
import os
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from accelerate import Accelerator

from chadgpt import GPT_CONFIG_250M, GPTModel, RoPEEmbedding, get_lr, generate
from chatml_tokenizer import get_chatml_tokenizer, IM_START_ID, IM_END_ID, PAD_ID, SFT_VOCAB_SIZE

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


'''
same architecture as phase 2 but with the vocab extended from 50257 to 50259 
to include the 2 new ChatML special tokens (<|im_start|> and <|im_end|>)
'''
SFT_MODEL_CONFIG = dict(GPT_CONFIG_250M)
SFT_MODEL_CONFIG["context_length"] = 4096
SFT_MODEL_CONFIG["vocab_size"] = SFT_VOCAB_SIZE


'''
patches RoPEEmbedding.forward so start_pos can be either:
  - an int (legacy path used during generation with KV cache)
  - a 1D tensor of explicit per-token positions (used during SFT training 
    where positions reset to 0 at each packed document boundary)

both paths are scaled by pi_scale to stay consistent with the phase 2 checkpoint 
which was trained under PI-scaled RoPE angles (raw position * 0.25)
without this scaling every token would look like it's at 4x its actual position 
from the model's calibrated point of view
'''
def enable_position_aware_rope(pi_scale=1.0):
    def _rope_forward(self, x, start_pos=0):
        seq_len = x.shape[-2]
        if torch.is_tensor(start_pos):
            t = start_pos.to(device=x.device, dtype=torch.float32).view(-1, 1)
            assert t.shape[0] == seq_len, (
                f"position_ids length {t.shape[0]} != seq_len {seq_len}"
            )
            t = t * pi_scale
        else:
            t = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(1).float()
            t = t * pi_scale
        freqs = (t * self.theta).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        return x * cos + self._rotate(x) * sin

    RoPEEmbedding.forward = _rope_forward


'''
grows the embedding table from 50257 to 50259 to make room for the 2 new ChatML tokens
copies existing rows and randomly initializes the new ones with std=0.02

since GPTModel.forward computes logits as x @ self.tok_emb.weight.T (weight-tied), 
resizing tok_emb alone handles both the input embedding and the output projection
'''
def resize_embedding_for_chatml(model, new_vocab_size):
    old_emb = model.tok_emb
    old_vocab_size, emb_dim = old_emb.weight.shape
    if old_vocab_size == new_vocab_size:
        return model
    assert new_vocab_size > old_vocab_size, "can only grow the vocab, not shrink it"

    new_emb = nn.Embedding(new_vocab_size, emb_dim).to(old_emb.weight.device)
    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = old_emb.weight
        nn.init.normal_(new_emb.weight[old_vocab_size:], mean=0.0, std=0.02)

    model.tok_emb = new_emb
    model.out_head = model.tok_emb.weight
    return model


'''
gradient-checkpointed forward that correctly passes position_ids through 

chadgpt_finetune.py's GPTModelFT hardcodes start_pos=0 inside its checkpointed block_fn 
which was fine for phase 2 (every example starts at position 0) but would silently discard 
our position_ids tensor during SFT training — this version closures the real start_pos through instead
'''
class GPTModelSFT(GPTModel):
    def forward(self, in_idx, past_key_values=None, start_pos=0):
        if not self.training or past_key_values is not None:
            return super().forward(in_idx, past_key_values=past_key_values, start_pos=start_pos)

        tok_embeds = self.tok_emb(in_idx)
        x = self.drop_emb(tok_embeds)

        for block in self.trf_blocks:
            def block_fn(x, _block=block, _start_pos=start_pos):
                out, _ = _block(x, past_kv=None, start_pos=_start_pos)
                return out
            x = grad_checkpoint(block_fn, x, use_reentrant=False)

        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        return logits, None


'''
computes cross entropy loss but only on the tokens where loss_mask = 1 (assistant turns)
system and user turn tokens have loss_mask = 0 so they don't contribute to the gradient
without this the model would just be doing continued pretraining on ChatML-formatted text 
instead of actual instruction tuning where only the response matters
'''
def masked_cross_entropy(logits, y, loss_mask, ignore_index=PAD_ID):
    per_token_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1),
        ignore_index=ignore_index, reduction="none",
    )
    mask_flat = loss_mask.reshape(-1)
    denom = mask_flat.sum().clamp(min=1.0)
    return (per_token_loss * mask_flat).sum() / denom


'''
reads the (tokens, loss_mask, positions) triplet shards produced by prepare_openhermes_sft.py
each row is CTX_LEN+1 tokens long so we can slice x = tokens[:-1], y = tokens[1:] 
just like ShardDataset does for pretraining
loss_mask is aligned with y (the targets) and position_ids with x (the inputs)
'''
class SFTShardDataset(Dataset):
    def __init__(self, shard_dir, shard_ids, prefix="sft_train_shard"):
        super().__init__()
        self.tokens, self.masks, self.positions = [], [], []
        row_counts = []

        for i in shard_ids:
            t = np.load(os.path.join(shard_dir, f"{prefix}_tokens_{i}.npy"), mmap_mode="r")
            m = np.load(os.path.join(shard_dir, f"{prefix}_lossmask_{i}.npy"), mmap_mode="r")
            p = np.load(os.path.join(shard_dir, f"{prefix}_positions_{i}.npy"), mmap_mode="r")
            assert t.shape[0] == m.shape[0] == p.shape[0], f"shard {i} array length mismatch"
            self.tokens.append(t)
            self.masks.append(m)
            self.positions.append(p)
            row_counts.append(t.shape[0])

        self.cumulative = np.cumsum([0] + row_counts)
        print(f"  {len(shard_ids)} SFT shard(s) -> {int(self.cumulative[-1]):,} packed sequences")

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.cumulative[1:], idx, side="right"))
        local_idx = idx - self.cumulative[shard_idx]

        tokens = np.asarray(self.tokens[shard_idx][local_idx], dtype=np.int64)
        mask = np.asarray(self.masks[shard_idx][local_idx], dtype=np.float32)
        positions = np.asarray(self.positions[shard_idx][local_idx], dtype=np.float32)

        x = torch.from_numpy(tokens[:-1].copy())
        y = torch.from_numpy(tokens[1:].copy())
        loss_mask = torch.from_numpy(mask[1:].copy())
        pos_ids = torch.from_numpy(positions[:-1].copy())

        return x, y, loss_mask, pos_ids


def get_sft_shard_ids(shard_dir, prefix="sft_train_shard"):
    ids = []
    i = 0
    while os.path.exists(os.path.join(shard_dir, f"{prefix}_tokens_{i}.npy")):
        ids.append(i)
        i += 1
    return ids


@torch.no_grad()
def evaluate_sft(model, val_loader, cfg):
    model.eval()
    device = next(model.parameters()).device
    losses = []
    for i, (x, y, loss_mask, pos_ids) in enumerate(val_loader):
        if i >= cfg["eval_steps"]:
            break
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        pos_ids_1d = pos_ids[0].to(device)
        logits, _ = model(x, start_pos=pos_ids_1d)
        loss = masked_cross_entropy(logits, y, loss_mask)
        losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


'''
generates a sample response during training to see how the model is doing 
wraps the prompt in ChatML format and generates until <|im_end|> is produced
'''
def sample_sft(model, tokenizer, device, prompt="Write a Python function that checks if a string is a palindrome."):
    model.eval()
    with torch.no_grad():
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(
            chat_prompt, allowed_special={"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
        )
        idx = torch.tensor(ids).unsqueeze(0).to(device)
        out = generate(model, idx, max_new_tokens=80, temperature=0.8, top_k=40, eos_id=IM_END_ID)
    model.train()
    return tokenizer.decode(out.squeeze(0).tolist())


'''
SFT training config 

gentler learning rate than pretraining (1.5e-5 vs 3e-4) — standard SFT range
batch_size=1 because context_length=4096, same constraint as phase 2

with ~8,888 packed rows and grad_accum=64 that's ~139 steps per epoch
max_steps=350 targets ~2.5 epochs which is the standard 2-3 epoch range 
for SFT to avoid overfitting a 250M model on this size dataset

PI scaling must match phase 2 (1024/4096 = 0.25) to keep RoPE angles consistent
'''
SFT_CONFIG = {
    "shard_dir": "/kaggle/working",
    "shard_prefix": "sft_train_shard",
    "val_shard_count": 1,

    "context_length": 4096,

    "lr": 1.5e-5,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "grad_clip": 1.0,

    "warmup_steps": 15,
    "max_steps": 350,
    "min_lr_ratio": 0.1,

    "batch_size": 1,
    "grad_accum_steps": 64,

    "eval_interval": 25,
    "eval_steps": 20,
    "log_interval": 10,

    "use_position_interpolation": True,
    "pi_scale": 0.25,

    "gradient_checkpointing": True,

    "ckpt_dir": "/kaggle/working/checkpoints",
    "save_interval": 25,

    "init_from": "checkpoint.pt",
    "init_phase_required": 2,
}


def train_sft(cfg):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"[SFT] Using {accelerator.num_processes} GPU(s) — device: {device}")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    pi_scale = cfg["pi_scale"] if cfg.get("use_position_interpolation", True) else 1.0
    enable_position_aware_rope(pi_scale=pi_scale)
    accelerator.print(f"RoPE position patch enabled, pi_scale={pi_scale}")

    '''
    model is built at the BASE vocab size (50257) so the phase 2 checkpoint's tok_emb 
    loads without a shape mismatch — we resize to 50259 AFTER loading weights, not before
    '''
    model_cls = GPTModelSFT if cfg.get("gradient_checkpointing", True) else GPTModel
    base_init_cfg = dict(GPT_CONFIG_250M)
    base_init_cfg["context_length"] = SFT_MODEL_CONFIG["context_length"]
    model = model_cls(base_init_cfg)
    accelerator.print(
        f"Model built at base vocab_size={GPT_CONFIG_250M['vocab_size']} "
        f"(resizing to {SFT_VOCAB_SIZE} after checkpoint load, before optimizer + accelerator.prepare)"
    )

    all_shard_ids = get_sft_shard_ids(cfg["shard_dir"], cfg["shard_prefix"])
    if not all_shard_ids:
        raise FileNotFoundError(
            f"No SFT shards found in {cfg['shard_dir']} with prefix '{cfg['shard_prefix']}'. "
            f"Run prepare_openhermes_sft.py first."
        )
    val_count = min(cfg["val_shard_count"], len(all_shard_ids) - 1) if len(all_shard_ids) > 1 else 0
    train_shard_ids = all_shard_ids[: len(all_shard_ids) - val_count] if val_count else all_shard_ids
    val_shard_ids = all_shard_ids[len(all_shard_ids) - val_count:] if val_count else all_shard_ids[-1:]

    accelerator.print(f"Train shards: {train_shard_ids}")
    train_ds = SFTShardDataset(cfg["shard_dir"], train_shard_ids, cfg["shard_prefix"])
    accelerator.print(f"Val shards: {val_shard_ids}")
    val_ds = SFTShardDataset(cfg["shard_dir"], val_shard_ids, cfg["shard_prefix"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False, drop_last=True,
    )

    ckpt_path = os.path.join(cfg["ckpt_dir"], "sft_checkpoint.pt")
    init_path = os.path.join(cfg["ckpt_dir"], cfg["init_from"])

    sft_step, base_tokens_seen, sft_tokens_seen = 0, 0, 0
    resumed_sft = False
    saved_optimizer_state = None
    saved_scaler_state = None

    '''
    checkpoint loading — happens on the raw model before accelerator.prepare()
    if we find a previous SFT checkpoint, resume from it (embedding was already resized in that run)
    otherwise load phase 2 weights at the original 50257 vocab THEN resize to 50259
    resizing before loading would cause a shape mismatch crash
    '''
    if os.path.exists(ckpt_path):
        probe = torch.load(ckpt_path, map_location="cpu")
        if probe.get("phase") == "sft":
            accelerator.print(f"Resuming SFT from sft_checkpoint.pt at step {probe['step']}")
            resize_embedding_for_chatml(model, SFT_VOCAB_SIZE)
            model.load_state_dict(probe["model"])
            saved_optimizer_state = probe["optimizer"]
            saved_scaler_state = probe.get("scaler")
            sft_step = probe["step"]
            sft_tokens_seen = probe.get("sft_tokens_seen", 0)
            base_tokens_seen = probe.get("base_tokens_seen", 0)
            resumed_sft = True
        del probe

    if not resumed_sft:
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"Phase 2 checkpoint not found at {init_path}. SFT needs a completed "
                f"Phase 2 run to initialize from."
            )
        accelerator.print(f"Initializing SFT from: {init_path}")
        base_ckpt = torch.load(init_path, map_location="cpu")
        if cfg.get("init_phase_required") is not None:
            found_phase = base_ckpt.get("phase")
            if found_phase != cfg["init_phase_required"]:
                accelerator.print(
                    f"WARNING: expected phase={cfg['init_phase_required']} checkpoint, "
                    f"found phase={found_phase}. Proceeding anyway."
                )
        model.load_state_dict(base_ckpt["model"])
        resize_embedding_for_chatml(model, SFT_VOCAB_SIZE)
        base_tokens_seen = base_ckpt.get("tokens_seen", 0)
        accelerator.print(
            f"  → loaded Phase 2 weights ({base_tokens_seen/1e9:.3f}B tokens seen), "
            f"resized embedding to vocab_size={SFT_VOCAB_SIZE}. Optimizer fresh for SFT."
        )
        del base_ckpt

    '''
    optimizer must be built AFTER the embedding resize so it captures the grown tok_emb parameter
    building it before the resize would silently exclude the new ChatML token embeddings 
    from optimization — those 2 tokens would then never learn anything
    '''
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=cfg["betas"])

    if resumed_sft and saved_optimizer_state is not None:
        optimizer.load_state_dict(saved_optimizer_state)

    '''
    accelerator.prepare() also must come after the resize — with 2 GPUs this wraps the model 
    in DistributedDataParallel which registers gradient sync buckets from model.parameters() 
    at wrap time — swapping tok_emb after this would leave the new rows outside DDP's buckets
    '''
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    if resumed_sft and saved_scaler_state is not None and getattr(accelerator, "scaler", None) is not None:
        accelerator.scaler.load_state_dict(saved_scaler_state)

    tokenizer = get_chatml_tokenizer()
    train_iter = iter(train_loader)
    grad_accum = cfg["grad_accum_steps"]
    t0 = time.time()
    tokens_this_interval = 0
    model.train()

    while sft_step < cfg["max_steps"]:
        lr = get_lr(sft_step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y, loss_mask, pos_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, loss_mask, pos_ids = next(train_iter)

            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            pos_ids_1d = pos_ids[0].to(device)  # batch_size=1

            toks = x.numel() * accelerator.num_processes
            sft_tokens_seen += toks
            tokens_this_interval += toks
            is_last_micro = (micro_step == grad_accum - 1)

            sync_context = accelerator.no_sync(model) if not is_last_micro else nullcontext()
            with sync_context:
                logits, _ = model(x, start_pos=pos_ids_1d)
                loss = masked_cross_entropy(logits, y, loss_mask)
                scaled_loss = loss / grad_accum
                accelerator.backward(scaled_loss)

            loss_reduced = accelerator.reduce(loss.detach().clone(), reduction="mean")
            accum_loss += loss_reduced.item() / grad_accum

        accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        sft_step += 1

        if accelerator.is_main_process:
            if sft_step % cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                tok_per_sec = tokens_this_interval / elapsed if elapsed > 0 else 0
                total_tokens = base_tokens_seen + sft_tokens_seen
                print(
                    f"[SFT] step {sft_step:>5}/{cfg['max_steps']} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_per_sec:,.0f} | "
                    f"sft_tokens {sft_tokens_seen/1e6:.1f}M | total_tokens {total_tokens/1e9:.3f}B",
                    flush=True,
                )
                t0 = time.time()
                tokens_this_interval = 0

            if sft_step % cfg["eval_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                val_loss, val_ppl = evaluate_sft(raw_model, eval_loader, cfg)
                print(f"\n[SFT EVAL @ step {sft_step}] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
                print(f"[SAMPLE] {sample_sft(raw_model, tokenizer, device)}\n", flush=True)
                t0 = time.time()
                tokens_this_interval = 0

            if sft_step % cfg["save_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                ckpt = {
                    "phase": "sft",
                    "step": sft_step,
                    "base_tokens_seen": base_tokens_seen,
                    "sft_tokens_seen": sft_tokens_seen,
                    "tokens_seen": base_tokens_seen + sft_tokens_seen,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_cfg": cfg,
                    "model_cfg": SFT_MODEL_CONFIG,
                }
                if getattr(accelerator, "scaler", None) is not None:
                    ckpt["scaler"] = accelerator.scaler.state_dict()
                torch.save(ckpt, ckpt_path)
                print(f"  → [SFT] Saved sft_checkpoint.pt at step {sft_step}", flush=True)

    accelerator.wait_for_everyone()
    accelerator.print("SFT complete.")


if __name__ == "__main__":
    train_sft(SFT_CONFIG)