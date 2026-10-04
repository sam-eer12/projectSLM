"""Browser playground for the custom ChadGPT PyTorch model."""

import os
import sys
import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download


MODEL_REPO = "sam-eer12/chadGPT"
MODEL_REVISION = "1d79625c1b3194140ccfcb933ae49b6599820fbb"
DEFAULT_SYSTEM = "You are a helpful assistant."

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def get_runtime():
    local_dir = os.environ.get("CHADGPT_MODEL_DIR")
    model_dir = Path(local_dir) if local_dir else Path(snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=["model.safetensors", "config.json", "tokenizer.tiktoken",
                        "tokenizer_config.json", "inference.py", "modeling_chadgpt.py"],
    ))
    sys.path.insert(0, str(model_dir.resolve()))
    from inference import chat, load_model
    model, tokenizer = load_model(model_dir, device=os.environ.get("CHADGPT_DEVICE", "auto"))
    return model, tokenizer, chat


def text_content(content):
    """Accept both string messages and Gradio 6 text content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block["text"] for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    raise gr.Error("This playground supports text messages only.")


def respond(message, history, system_prompt, max_new_tokens, temperature, top_k):
    if not message.strip():
        raise gr.Error("Enter a message to start chatting.")
    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        if item["role"] in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": text_content(item["content"])})
    messages.append({"role": "user", "content": message})
    model, tokenizer, chat = get_runtime()
    try:
        result = chat(model, tokenizer, messages=messages,
                      max_new_tokens=int(max_new_tokens), temperature=float(temperature),
                      top_k=int(top_k))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return result or "The model ended its response without generating text. Try rephrasing your message."


def build_demo():
    with gr.Blocks(title="ChadGPT Playground", analytics_enabled=False) as demo:
        gr.Markdown(
            "# ChadGPT Playground\n"
            "Chat with a **250M-parameter language model trained from scratch on Kaggle** "
            "over approximately two months. The model runs in this demo session.\n\n"
            "[Model card & training details](https://huggingface.co/sam-eer12/chadGPT) · "
            "[Training notebook](https://huggingface.co/sam-eer12/chadGPT/blob/main/chadgpt.ipynb)\n\n"
            "This is an experimental educational model. Answers and generated code can be incorrect. "
            "Generation takes a moment; start with a short question."
        )
        with gr.Accordion("Generation settings", open=False):
            system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM, lines=2)
            max_new_tokens = gr.Slider(16, 256, value=96, step=16, label="Maximum output tokens")
            temperature = gr.Slider(0, 1.5, value=0.7, step=0.05,
                                    label="Temperature", info="0 uses greedy decoding.")
            top_k = gr.Slider(0, 100, value=40, step=1,
                              label="Top-k", info="0 disables top-k filtering.")
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=460, placeholder="Ask a question to try ChadGPT."),
            textbox=gr.Textbox(placeholder="Message ChadGPT…", max_lines=6),
            additional_inputs=[system_prompt, max_new_tokens, temperature, top_k],
            examples=[
                ["Explain what a neural network is in simple terms.", DEFAULT_SYSTEM, 96, 0.7, 40],
                ["Write a Python function that adds two numbers.", DEFAULT_SYSTEM, 96, 0.7, 40],
                ["Write a short story about a curious robot.", DEFAULT_SYSTEM, 128, 0.8, 40],
            ],
            cache_examples=False,
            run_examples_on_click=False,
            flagging_mode="never",
            save_history=False,
            concurrency_limit=1,
            api_name="chat",
        )
        gr.Markdown("The full conversation and output share a 4,096-token window. Clear the chat to start over.")
    return demo.queue(max_size=20, default_concurrency_limit=1)


demo = build_demo()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="Create a temporary public gradio.live link.")
    args = parser.parse_args()
    get_runtime()
    demo.launch(share=args.share)
