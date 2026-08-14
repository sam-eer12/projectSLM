
import tiktoken

'''
ChadGPT's base tokenizer is tiktoken's gpt2 encoding (vocab_size=50257, <|endoftext|>=50256)
for SFT we need 2 new special tokens for ChatML formatting:
    <|im_start|> = 50257
    <|im_end|>   = 50258

we extend the same gpt2 encoding rather than swapping in a different tokenizer 
so the SFT model can resume from the phase 2 checkpoint's embedding table without issues

we also reuse <|endoftext|> (50256) as the pad token — it's always loss_mask=0 
so it never contributes to the gradient
'''

IM_START_ID = 50257
IM_END_ID = 50258
PAD_ID = 50256
SFT_VOCAB_SIZE = 50259  # base 50257 + 2 new special tokens


'''
returns a tiktoken encoding identical to gpt2 but with <|im_start|> and <|im_end|> 
registered as additional special tokens so they get their own token ids instead of 
being broken up into subword pieces
'''
def get_chatml_tokenizer():
    base = tiktoken.get_encoding("gpt2")
    special_tokens = dict(base._special_tokens)
    special_tokens["<|im_start|>"] = IM_START_ID
    special_tokens["<|im_end|>"] = IM_END_ID

    enc = tiktoken.Encoding(
        name="gpt2_chatml",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )
    return enc


if __name__ == "__main__":
    enc = get_chatml_tokenizer()
    test = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"
    ids = enc.encode(test, allowed_special={"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
    print("Encoded:", ids)
    print("Decoded:", enc.decode(ids))
    assert enc.decode(ids) == test
    print("Round-trip OK. Vocab size needed:", SFT_VOCAB_SIZE)
