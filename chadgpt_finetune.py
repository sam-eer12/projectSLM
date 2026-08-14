
import os
import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from accelerate import Accelerator
import tiktoken

from chadgpt import (
    GPT_CONFIG_250M,
    GPTModel,
    RoPEEmbedding,
    ShardDataset,
    get_lr,
    evaluate,
    sample,
)


'''
same as get_shard_paths in chadgpt.py but takes an explicit list of shard ids 
instead of a start+count range because phase 2 uses specific shards not a contiguous block
'''
def shard_paths_from_ids(shard_dir, shard_ids):
    paths = []
    for i in shard_ids:
        p = os.path.join(shard_dir, f"slm_train_shard_{i}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Shard not found: {p}")
        paths.append(p)
    return paths

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


'''
same architecture as phase 1 just with a longer context length of 4096
RoPE is computed dynamically from position so no new parameters are added
'''
FT_MODEL_CONFIG = dict(GPT_CONFIG_250M)
FT_MODEL_CONFIG["context_length"] = 4096


'''
position interpolation (PI) for RoPE 

phase 1 only trained on positions 0-1023 so at context length 4096 the positions 1024-4095 
would hit rotation angles way outside what the model saw during pretraining
PI rescales the position index so the full 0-4095 range maps back into 0-1023 
this way the model doesn't need to learn new frequency patterns from scratch 

we patch it onto RoPEEmbedding here instead of editing chadgpt.py so phase 1's code stays untouched
'''
def apply_position_interpolation(scale):
    def _rope_forward_pi(self, x, start_pos=0):
        seq_len = x.shape[-2]
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(1) * scale
        freqs = (t * self.theta).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        return x * cos + self._rotate(x) * sin

    RoPEEmbedding.forward = _rope_forward_pi


'''
gradient checkpointed version of GPTModel for memory safety

same architecture and state_dict as phase 1 — only the forward pass changes
during training each transformer block runs under torch.utils.checkpoint so the internal 
activations (attention scores etc) get discarded after the block and recomputed on backward 
this is what makes context_length=4096 with batch_size=1 fit on a 16GB T4

without this the stored activations would eat ~2.5GB+ per GPU
falls back to normal forward when not training or when KV cache is being used for generation
'''
class GPTModelFT(GPTModel):
    def forward(self, in_idx, past_key_values=None, start_pos=0):
        if not self.training or past_key_values is not None:
            return super().forward(in_idx, past_key_values=past_key_values, start_pos=start_pos)

        tok_embeds = self.tok_emb(in_idx)
        x = self.drop_emb(tok_embeds)

        for block in self.trf_blocks:
            def block_fn(x, _block=block):
                out, _ = _block(x, past_kv=None, start_pos=0)
                return out
            x = grad_checkpoint(block_fn, x, use_reentrant=False)

        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        return logits, None


'''
phase 2 fine-tuning config 

uses a different set of shards than phase 1 (shards 41-43 instead of 0-39)
lower learning rate (3e-5 vs 3e-4) since we're fine-tuning not training from scratch
batch_size=1 because context_length=4096 is 4x longer so each sample is 4x bigger

checkpoint policy:
  - phase 1's latest.pt is read once at the start and never written to — it's your rollback point
  - all phase 2 progress goes to checkpoint.pt only, tagged with phase=2
  - to restart phase 2 from scratch just delete checkpoint.pt and rerun
'''
FT_CONFIG = {
    "shard_dir":        ".",
    "train_shard_ids":  [41, 42],
    "val_shard_ids":    [43],
    "context_length":   4096,

    "lr":               3e-5,
    "weight_decay":     0.1,
    "betas":            (0.9, 0.95),
    "grad_clip":        1.0,

    "warmup_steps":     20,
    "max_steps":        500,
    "min_lr_ratio":      0.1,

    "batch_size":        1,
    "grad_accum_steps": 64,

    "eval_interval":    50,
    "eval_steps":        10,
    "log_interval":       5,

    "use_position_interpolation": True,
    "pi_base_context":            1024,

    "gradient_checkpointing": True,

    "ckpt_dir":          "/kaggle/working/checkpoints",
    "save_interval":     50,

    "init_from":          "latest.pt",
}


def train_finetune(cfg):

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"[Phase 2] Using {accelerator.num_processes} GPU(s) — device: {device}")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    if cfg.get("use_position_interpolation", True):
        pi_scale = cfg["pi_base_context"] / cfg["context_length"]
        apply_position_interpolation(pi_scale)
        accelerator.print(
            f"Position Interpolation enabled: scaling RoPE positions by {pi_scale:.4f} "
            f"({cfg['context_length']} → {cfg['pi_base_context']} effective range)"
        )
    else:
        accelerator.print("Position Interpolation disabled — using raw (unscaled) RoPE positions.")

    model_cls = GPTModelFT if cfg.get("gradient_checkpointing", True) else GPTModel
    model = model_cls(FT_MODEL_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"Total parameters: {total_params:,} | context_length={FT_MODEL_CONFIG['context_length']} | "
        f"gradient_checkpointing={cfg.get('gradient_checkpointing', True)}"
    )

    '''
    fresh optimizer for phase 2 — we reset it because we're switching to a different lr regime
    weight decay only on 2D+ params (matrices), not on biases or layer norm scales
    '''
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=cfg["betas"])

    train_paths = shard_paths_from_ids(cfg["shard_dir"], cfg["train_shard_ids"])
    val_paths   = shard_paths_from_ids(cfg["shard_dir"], cfg["val_shard_ids"])

    accelerator.print("Train shards:")
    train_ds = ShardDataset(train_paths, cfg["context_length"])
    accelerator.print("Val shards:")
    val_ds   = ShardDataset(val_paths,   cfg["context_length"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True, drop_last=True,
    )

    '''
    eval_loader is not passed through accelerator.prepare() to avoid DDP rank mismatch 
    during evaluate() — same approach as phase 1
    '''
    eval_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    tokenizer  = tiktoken.get_encoding("gpt2")
    train_iter = iter(train_loader)
    grad_accum = cfg["grad_accum_steps"]
    t0                   = time.time()
    tokens_this_interval = 0
    model.train()

    ckpt_path   = os.path.join(cfg["ckpt_dir"], "checkpoint.pt")
    latest_path = os.path.join(cfg["ckpt_dir"], cfg["init_from"])

    ft_step, base_tokens_seen, ft_tokens_seen = 0, 0, 0
    resumed_phase2 = False

    '''
    resume logic:
    if checkpoint.pt has an in-progress phase 2 run, resume from it 
    otherwise load weights from phase 1's latest.pt (weights only, optimizer is fresh)
    latest.pt is only ever read here, never written to by this script
    also saves/restores the fp16 grad scaler state to avoid instability when resuming
    '''
    if os.path.exists(ckpt_path):
        probe = torch.load(ckpt_path, map_location="cpu")
        if probe.get("phase") == 2:
            accelerator.print(f"Resuming Phase 2 fine-tune from checkpoint.pt at step {probe['step']}")
            accelerator.unwrap_model(model).load_state_dict(probe["model"])
            optimizer.load_state_dict(probe["optimizer"])
            if getattr(accelerator, "scaler", None) is not None and "scaler" in probe:
                accelerator.scaler.load_state_dict(probe["scaler"])
            ft_step          = probe["step"]
            ft_tokens_seen   = probe.get("ft_tokens_seen", 0)
            base_tokens_seen = probe.get("base_tokens_seen", 0)
            resumed_phase2   = True
        del probe

    if not resumed_phase2:
        if not os.path.exists(latest_path):
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found at {latest_path}. Fine-tuning "
                f"needs a Phase 1 run to initialize from before Phase 2 can start."
            )
        accelerator.print(f"Initializing Phase 2 from Phase 1 checkpoint: {latest_path}")
        base_ckpt = torch.load(latest_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(base_ckpt["model"])
        base_tokens_seen = base_ckpt.get("tokens_seen", 0)
        accelerator.print(
            f"  → loaded weights from Phase 1 step {base_ckpt.get('step')}, "
            f"{base_tokens_seen/1e9:.3f}B tokens seen. Optimizer reset for Phase 2."
        )
        del base_ckpt

    while ft_step < cfg["max_steps"]:
        lr = get_lr(ft_step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            toks                  = x.numel() * accelerator.num_processes
            ft_tokens_seen       += toks
            tokens_this_interval += toks
            is_last_micro         = (micro_step == grad_accum - 1)

            '''
            only sync gradients across GPUs on the last micro step of each accumulation batch 
            skipping sync on earlier micro steps saves a lot of communication overhead 
            '''
            sync_context = accelerator.no_sync(model) if not is_last_micro else nullcontext()
            with sync_context:
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=50256
                )
                scaled_loss = loss / grad_accum
                accelerator.backward(scaled_loss)

            loss_reduced = accelerator.reduce(loss.detach().clone(), reduction="mean")
            accum_loss  += loss_reduced.item() / grad_accum

        accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        ft_step += 1

        if accelerator.is_main_process:
            if ft_step % cfg["log_interval"] == 0:
                elapsed      = time.time() - t0
                tok_per_sec  = tokens_this_interval / elapsed if elapsed > 0 else 0
                total_tokens = base_tokens_seen + ft_tokens_seen
                print(
                    f"[FT] step {ft_step:>4}/{cfg['max_steps']} | loss {accum_loss:.4f} | "
                    f"ppl {math.exp(min(accum_loss, 20)):.1f} | lr {lr:.2e} | "
                    f"tok/s {tok_per_sec:,.0f} | ft_tokens {ft_tokens_seen/1e6:.1f}M | "
                    f"total_tokens {total_tokens/1e9:.3f}B",
                    flush=True,
                )
                t0                   = time.time()
                tokens_this_interval = 0

            if ft_step % cfg["eval_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                val_loss, val_ppl = evaluate(raw_model, eval_loader, cfg)
                print(f"\n[FT EVAL @ step {ft_step}] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
                print(f"[SAMPLE] {sample(raw_model, tokenizer, device)}\n", flush=True)
                t0                   = time.time()
                tokens_this_interval = 0

            if ft_step % cfg["save_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                ckpt = {
                    "phase":            2,
                    "step":             ft_step,
                    "base_tokens_seen": base_tokens_seen,
                    "ft_tokens_seen":   ft_tokens_seen,
                    "tokens_seen":      base_tokens_seen + ft_tokens_seen,
                    "model":            raw_model.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "train_cfg":        cfg,
                    "model_cfg":        FT_MODEL_CONFIG,
                }
                if getattr(accelerator, "scaler", None) is not None:
                    ckpt["scaler"] = accelerator.scaler.state_dict()

                # only checkpoint.pt is written — latest.pt (phase 1) stays untouched
                torch.save(ckpt, ckpt_path)
                print(f"  → [FT] Saved checkpoint.pt at step {ft_step} (latest.pt untouched)", flush=True)

    accelerator.wait_for_everyone()
    accelerator.print("Phase 2 fine-tuning complete.")


if __name__ == "__main__":
    train_finetune(FT_CONFIG)
