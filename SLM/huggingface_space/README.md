---
title: ChadGPT Playground
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: '3.11'
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
license: mit
short_description: Chat with ChadGPT, a 250M model trained from scratch on Kaggle
models:
- sam-eer12/chadGPT
---

# ChadGPT Playground

A browser chat interface for [sam-eer12/chadGPT](https://huggingface.co/sam-eer12/chadGPT), a custom 250M-parameter language model trained from scratch on Kaggle over approximately two months.

On Hugging Face Spaces, open the **App** tab, enter a message, and send it. Generation settings expose the system prompt, temperature, top-k, and output-token limit. Conversation history is passed to the model for multi-turn chat. Visitors do not need local installations or tokens.

The Space configuration uses CPU Basic hardware. The loader selects CUDA when a notebook GPU is available, otherwise Apple MPS or CPU. The app handles one generation request at a time with a maximum of 256 new tokens per response. It downloads approximately 1 GB of model weights once during startup. A sleeping Space may take time to wake up. The weights and inference code are pinned to the verified initial model release.

The model is educational and experimental: it can produce inaccurate answers, repetitive text, and incorrect code. See the model card for architecture, datasets, training details, and limitations.

## Inference in a Colab or Kaggle notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/sam-eer12/chadGPT/colab)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://huggingface.co/sam-eer12/chadGPT/kaggle)

Open either launcher link and sign in to that platform. Run **Install dependencies → Download and load → Test inference**. The notebook installs the small inference dependencies and automatically fetches the standalone inference code, tokenizer, configuration, and model weights from Hugging Face. There is no repository to clone or file to upload. A [downloadable copy](https://huggingface.co/sam-eer12/chadGPT/resolve/main/chadgpt_playground.ipynb?download=true) is also available.

Responses print directly in the notebook. Change the prompt and rerun the test cell to ask another question.

For Kaggle, turn **Internet** on in the notebook settings and optionally select a GPU accelerator. Then run the cells in order. If the account cannot enable internet or a GPU, complete the verification requested by Kaggle on its website. The launcher also works on CPU.

For Colab, connect a runtime and optionally select a GPU before running the cells. Generation runs in your notebook runtime. No Hugging Face PRO subscription or model-download token is needed.

## Run the Gradio app locally

To run a temporary browser demo on your own computer:

```bash
python -m pip install -r SLM/huggingface_space/requirements-notebook.txt
python SLM/huggingface_space/app.py --share
```

Install PyTorch first if it is not already available. Keep the terminal process running while using the public link. Do not use the Space-only `requirements.txt` in a GPU notebook: it intentionally installs a CPU PyTorch wheel.

## Create this playground on Hugging Face

1. Visit [Create a new Space](https://huggingface.co/new-space).
2. Choose your owner, enter a Space name such as `chadGPT-playground`, select **Gradio**, and choose **Public** visibility with **CPU Basic** hardware.
3. Upload `app.py`, `requirements.txt`, and this `README.md` to the Space's root folder using **Files → Add file → Upload files**. Retain the YAML header in `README.md`.
4. Hugging Face builds the app automatically. When the status becomes **Running**, open **App** and send a message.
5. Add the Space URL to your model card so visitors can find it. The `models` metadata in this README also associates the Space with `sam-eer12/chadGPT`.

No secret or inference-provider API key is needed because the model repository is public and the app executes its own PyTorch loader. If Hugging Face prompts for a paid plan to create a compute Space, account eligibility must be enabled on the website before this Gradio Space can run. CPU Basic itself has no hourly hardware charge; selecting paid GPU hardware would add a separate charge. See [Spaces account and hardware requirements](https://huggingface.co/docs/hub/spaces-overview).

To publish the same files with the HF CLI:

```bash
hf repos create sam-eer12/chadGPT-playground --repo-type space --space-sdk gradio --flavor cpu-basic --public
hf upload sam-eer12/chadGPT-playground SLM/huggingface_space . --repo-type space \
  --include app.py --include README.md --include requirements.txt --include LICENSE
```

Local development uses the same app. Install the model inference dependencies and `gradio==6.29.1`, then run `python SLM/huggingface_space/app.py`. Set `CHADGPT_MODEL_DIR` to an existing model export to skip downloading during development, or `CHADGPT_DEVICE=cpu` to explicitly select CPU.
