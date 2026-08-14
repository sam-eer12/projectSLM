
import os
import math
import time
from contextlib import nullcontext
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


GPT_CONFIG_250M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_kv_heads": 4,
    "n_layers": 18,
    "drop_rate": 0.0,
    "qkv_bias": False,
}



'''
 train config according to 5 billion dataset tokens 
 total steps  = (n_train_shards * no of tokens in 1 shard ) / effective batch size
 the model is trained on kaggle platform for their GPUs so the folder structure is according to that 
'''

TRAIN_CONFIG = {
    "shard_dir":       ".",
    "n_train_shards":  40,
    "n_val_shards":    10,
    "context_length":  1024,

    "lr":              3e-4,
    "weight_decay":    0.1,
    "betas":           (0.9, 0.95),
    "grad_clip":       1.0,

    "warmup_steps":    500,
    "max_steps":       15259,
    "min_lr_ratio":    0.1,

    "batch_size":      2,
    "grad_accum_steps": 64,

    "eval_interval":   100,
    "eval_steps":      20,
    "log_interval":    10,

    "ckpt_dir":        "/kaggle/working/checkpoints",
    "save_interval":   100,
}



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


'''
if you want you can also use swiglu in here instead of the gelu ,
to use this in the feed forward network just chnage the entire nn.sequential to     SwiGLU(cfg["emb_dim"])

class SwiGLU(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.w1 = nn.Linear(emb_dim, 4 * emb_dim, bias=False)
        self.w2 = nn.Linear(emb_dim, 4 * emb_dim, bias=False)
        self.w3 = nn.Linear(4 * emb_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
'''
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


''' 
it helps the model to expand the layers and then look at the minute details so it expands it to 4 times the embedding dimension,
but input and output dimension should be the same
'''

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


''' 
in this the theta is used to rotate the matrix for the Q and K matrices in 2d dimension to find the relative distance between them 
'''
class RoPEEmbedding(nn.Module):
    def __init__(self, emb_dim, n_heads, base=10000):
        super().__init__()
        self.head_dim = emb_dim // n_heads
        theta = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("theta", theta)

    def _rotate(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x, start_pos=0):
        seq_len = x.shape[-2]
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(1)
        freqs = (t * self.theta).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        return x * cos + self._rotate(x) * sin


''' 
here is also the implementation of the kv caching where we cache the past token in K and V 
and use the generate attn score to replace the Q 
which reduces the computational power and caches the K and V and also only 1 token is generate at a time on the attn output side

in the GQA part 
group size (n_rep) is n_heads / n_kv_heads 
this is the group of heads sharing same K and V values to reduce the KV cache while still preserving the output quality 
if you increase this value by either reducing kv heads or by increaseing attn heads memory usage will decrease but the output quality will decrease and 
if you decrease this value doing vice versa the memroy usage will increase but the output quality will also increase 
'''

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, n_kv_heads, qkv_bias=False):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert d_out % n_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = d_out // n_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)
        self.rope     = RoPEEmbedding(d_out, n_heads)

    def forward(self, x, past_kv=None, start_pos=0):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T,   self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        new_kv = (k, v)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if T > 1 else False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            n_heads=cfg["n_heads"],
            n_kv_heads=cfg["n_kv_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff           = FeedForward(cfg)
        self.norm1        = LayerNorm(cfg["emb_dim"])
        self.norm2        = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None, start_pos=0):
        shortcut = x
        x = self.norm1(x)
        x, new_kv = self.att(x, past_kv=past_kv, start_pos=start_pos)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, new_kv


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb  = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = self.tok_emb.weight

        self.apply(self._init_weights)
        for block in self.trf_blocks:
            nn.init.normal_(block.att.out_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * cfg["n_layers"]))
            nn.init.normal_(block.ff.layers[2].weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * cfg["n_layers"]))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx, past_key_values=None, start_pos=0):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = self.drop_emb(tok_embeds)

        new_key_values = []
        for i, block in enumerate(self.trf_blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, layer_kv = block(x, past_kv=layer_past, start_pos=start_pos)
            new_key_values.append(layer_kv)

        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        return logits, new_key_values



def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
    past_key_values = None
    start_pos = 0
    curr_idx = idx

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, past_key_values = model(curr_idx, past_key_values=past_key_values,
                                            start_pos=start_pos)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).any():
            break

        idx       = torch.cat((idx, idx_next), dim=1)
        start_pos += curr_idx.shape[1]
        curr_idx  = idx_next

    return idx



class ShardDataset(Dataset):
    def __init__(self, shard_paths: list[str], context_length: int):
        super().__init__()
        self.context_length = context_length
        self.mmaps          = []
        self.shard_lengths  = []

        for p in shard_paths:
            mm = np.memmap(p, dtype=np.uint16, mode='r')
            self.mmaps.append(mm)
            self.shard_lengths.append(len(mm))


        self.stride = context_length
        stride_lengths = [l // self.stride for l in self.shard_lengths]
        self.cumulative = np.cumsum([0] + stride_lengths)
        total_tokens = sum(self.shard_lengths)
        total_chunks = self.cumulative[-1]
        print(f"  {len(shard_paths)} shard(s) → {total_tokens:,} tokens "
              f"| {total_chunks:,} non-overlapping chunks (stride={self.stride})")

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):

        shard_idx = np.searchsorted(self.cumulative[1:], idx, side='right')
        local_chunk = idx - self.cumulative[shard_idx]
        local_idx   = local_chunk * self.stride   # token offset within shard

        end = local_idx + self.context_length + 1
        if end <= self.shard_lengths[shard_idx]:
            chunk = self.mmaps[shard_idx][local_idx:end]
        else:
            part1    = self.mmaps[shard_idx][local_idx:]
            overflow = end - self.shard_lengths[shard_idx]
            if shard_idx + 1 < len(self.mmaps):
                part2 = self.mmaps[shard_idx + 1][:overflow]
                chunk = np.concatenate([part1, part2])
            else:
                chunk = np.pad(part1, (0, overflow), constant_values=0)

        chunk = chunk.astype(np.int64)


        counts = np.bincount(chunk, minlength=50257)
        if counts.max() > len(chunk) * 0.5:

            x = torch.zeros(self.context_length, dtype=torch.long)
            y = torch.full((self.context_length,), 50256, dtype=torch.long)
            return x, y

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


def get_shard_paths(shard_dir, start, count):
    paths = []
    for i in range(start, start + count):
        p = os.path.join(shard_dir, f"slm_train_shard_{i}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Shard not found: {p}")
        paths.append(p)
    return paths



def get_lr(step, cfg):
    warmup    = cfg["warmup_steps"]
    max_steps = cfg["max_steps"]
    lr        = cfg["lr"]
    min_lr    = lr * cfg["min_lr_ratio"]

    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))



@torch.no_grad()
def evaluate(model, val_loader, cfg):

    model.eval()
    model_device = next(model.parameters()).device
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg["eval_steps"]:
            break
        x, y = x.to(model_device), y.to(model_device)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=50256
        )
        losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)


def sample(model, tokenizer, device, prompt="The model learns"):
    model.eval()
    with torch.no_grad():
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate(model, ids, max_new_tokens=30, temperature=0.8, top_k=40)
    model.train()
    return token_ids_to_text(out, tokenizer)



def train(cfg):

    accelerator = Accelerator(mixed_precision="fp16")
    device      = accelerator.device

    accelerator.print(f"Using {accelerator.num_processes} GPU(s) — device: {device}")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)


    model = GPTModel(GPT_CONFIG_250M)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Total parameters: {total_params:,}")


    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=cfg["betas"])


    train_paths = get_shard_paths(cfg["shard_dir"], 0,                   cfg["n_train_shards"])
    val_paths   = get_shard_paths(cfg["shard_dir"], cfg["n_train_shards"], cfg["n_val_shards"])

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

    eval_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
    )


    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    tokenizer            = tiktoken.get_encoding("gpt2")
    train_iter           = iter(train_loader)
    step, tokens_seen    = 0, 0
    grad_accum           = cfg["grad_accum_steps"]
    t0                   = time.time()
    tokens_this_interval = 0
    model.train()

    resume_path = os.path.join(cfg["ckpt_dir"], "latest.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        accelerator.print(f"Resumed from step {step}")
    else:
        accelerator.print("Starting fresh run")


    while step < cfg["max_steps"]:
        lr = get_lr(step, cfg)
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
            tokens_seen          += toks
            tokens_this_interval += toks
            is_last_micro         = (micro_step == grad_accum - 1)


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
        step += 1


        if accelerator.is_main_process:
            if step % cfg["log_interval"] == 0:
                elapsed     = time.time() - t0
                tok_per_sec = tokens_this_interval / elapsed if elapsed > 0 else 0
                print(
                    f"step {step:>5} | loss {accum_loss:.4f} | "
                    f"ppl {math.exp(min(accum_loss, 20)):.1f} | lr {lr:.2e} | "
                    f"tok/s {tok_per_sec:,.0f} | tokens {tokens_seen/1e9:.3f}B",
                    flush=True,
                )
                t0                   = time.time()
                tokens_this_interval = 0

            if step % cfg["eval_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                val_loss, val_ppl = evaluate(raw_model, eval_loader, cfg)
                print(f"\n[EVAL @ step {step}] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
                print(f"[SAMPLE] {sample(raw_model, tokenizer, device)}\n", flush=True)

                t0                   = time.time()
                tokens_this_interval = 0

            if step % cfg["save_interval"] == 0:
                raw_model = accelerator.unwrap_model(model)
                ckpt = {
                    "step":        step,
                    "tokens_seen": tokens_seen,
                    "model":       raw_model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "train_cfg":   cfg,
                    "model_cfg":   GPT_CONFIG_250M,
                }

                torch.save(ckpt, os.path.join(cfg["ckpt_dir"], "latest.pt"))
                torch.save(ckpt, os.path.join(cfg["ckpt_dir"], "checkpoint.pt"))
                print(f"  → Saved checkpoint at step {step}", flush=True)

    accelerator.wait_for_everyone()
    accelerator.print("Done.")



if __name__ == "__main__":
    train(TRAIN_CONFIG)
