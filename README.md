# OmniVoice Studio: Streamlined Voice Cloning & Prompt Management

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model License: CC BY--NC 4.0](https://img.shields.io/badge/Model%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio%204.x-orange.svg)](https://gradio.app/)

An enhanced, streamlined, and modular version of **OmniVoice**, featuring **persistent `.pt` voice prompt serialization**, clean English-only UI, custom offline caching, and optimized generation controls.

---

### 🌟 Powered by Hugging Face & Xiaomi AI Lab

<table>
  <tr>
    <td align="center" width="140px">
      <a href="https://huggingface.co/k2-fsa/OmniVoice">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="80px" alt="Hugging Face Logo"><br>
        <b>Hugging Face Model</b>
      </a>
    </td>
    <td>
      <b>OmniVoice: State-of-the-Art Zero-Shot Voice Cloning for 600+ Languages</b><br>
      Explore pre-trained checkpoints, interactive web demos, and official research releases on the Hugging Face Hub.<br>
      <a href="https://huggingface.co/k2-fsa/OmniVoice">📦 Model Card: <code>k2-fsa/OmniVoice</code></a> &nbsp;|&nbsp;
      <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice">🚀 Official HF Space Demo</a>
    </td>
  </tr>
</table>

</div>

---

## ⚖️ CRITICAL ETHICAL USE & LEGAL DISCLAIMER

> [!CAUTION]
> ### 🛑 STRICT PROHIBITION ON UNAUTHORIZED VOICE CLONING
> **Do NOT use this software, model weights, or generated audio for any purpose involving cloning the voice of any individual without their explicit, verifiable, prior written consent.**
> 
> By downloading, cloning, or operating this project, you strictly agree to the following terms:
> 1. **No Unauthorized Cloning:** You will not impersonate any person (living or deceased), public figure, celebrity, or private individual without valid legal consent.
> 2. **No Deceptive Content:** You will not generate defamatory, misleading, fraudulent, non-consensual sexual, or politically manipulative speech (Deepfakes).
> 3. **Non-Commercial Restriction:** The pre-trained neural network weights (`k2-fsa/OmniVoice`) are licensed strictly under **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)**. Commercial deployment, monetization, or paid API integration of the model weights is forbidden without explicit authorization from the original copyright holders.
> 
> The creators and maintainers of this repository assume **no liability** for misuse, illegal activities, or copyright/personality rights violations arising from the operation of this software.

---

## 🏆 Attribution & Acknowledgements

This project is built upon the foundational work of **OmniVoice** by the **Xiaomi AI Lab Next-gen Kaldi Team**. We give our utmost gratitude and full credit to the original authors and maintainers:

* **Original Open-Source Project:** [k2-fsa/OmniVoice on GitHub](https://github.com/k2-fsa/OmniVoice)
* **Original Authors & Research Team:** Han Zhu (`zhuhan`), Next-gen Kaldi team (`k2-fsa`), and contributors.
* **Original License:** Apache License 2.0 (Codebase) & CC-BY-NC 4.0 (Model Checkpoints).

All core neural network architectures, flow-matching mechanisms, and multi-lingual acoustic modeling belong to their pioneering research.

---

## ✨ Key Enhancements & Features

This repository introduces several critical engineering upgrades to make local deployment cleaner, faster, and more reusable:

### 1. 💾 Voice Prompt Serialization (`VoiceClonePrompt.save` & `.load`)
Instead of re-extracting speaker embeddings and processing reference audio on every single synthesis request, this enhanced version allows:
* **Saving (`.pt`):** Extract the acoustic prompt (`VoiceClonePrompt`) once from any reference audio and save it to a lightweight `.pt` file.
* **Instant Loading:** Directly upload or select a pre-saved `.pt` prompt file in the UI to skip reference audio processing entirely.
* **Automatic Export:** Every time you generate speech, the extracted prompt is automatically exported as a downloadable artifact in `saved_prompts/`.

### 2. 🎨 Clean, Streamlined English-Only UI
* Removed redundant dual-language Chinese/English clutter from labels and buttons for a clean, modern aesthetic.
* Removed the experimental "Voice Design" tab to focus 100% on **High-Fidelity Voice Cloning**.
* **Optimized Defaults:** Default generation speed is tuned to **`0.9`** for more natural, deliberate pacing and clearer articulation across complex sentences.

### 3. 🛡️ Offline Caching & Storage Management
* Automatic redirect of Hugging Face cache (`HF_HOME`, `HF_HUB_CACHE`) and PyTorch cache (`TORCH_HOME`) to a local `models_cache/` directory.
* Prevents `C:\` drive exhaustion when downloading the ~3.36 GB model weights.
* Pre-enabled `HF_HUB_OFFLINE=1` once weights are downloaded to eliminate startup latency and network timeouts.

---

## 🛠️ Installation & Setup

### Prerequisites
* Windows 10/11 or Linux with NVIDIA GPU (CUDA 11.8+ recommended).
* Python 3.9, 3.10, or 3.11.

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/OmniVoice-Studio.git
cd OmniVoice-Studio
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Studio locally

Simply launch the entry point script:
```powershell
python "Omni Voice.py"
```

1. On the first launch, the script will automatically download the `k2-fsa/OmniVoice` checkpoint (~3.36 GB) and store it safely in the `models_cache/` folder.
2. Once loaded, open your web browser to the Gradio interface:
   ```
   http://127.0.0.1:7860
   ```

---

## 📂 Project Structure

```text
OmniVoice-Studio/
├── Omni Voice.py          # Main application runner, offline cache config & Gradio server
├── requirements.txt       # Clean, exact Python dependencies
├── LICENSE                # Apache 2.0 License with Xiaomi AI Lab attribution
├── README.md              # Project documentation and legal disclaimer
└── omnivoice/             # Enhanced bundled package (self-contained)
    ├── __init__.py        # Exports OmniVoice, VoiceClonePrompt, generation configs
    ├── models/
    │   └── omnivoice.py   # Core model + VoiceClonePrompt serialization (.save/.load)
    ├── cli/
    │   └── demo.py        # Streamlined English-only Gradio UI (0.9 speed default)
    └── ...                # Core utility and generation modules
```

---

## 📜 License Summary

* **Source Code:** Distributed under the [Apache License 2.0](file:///LICENSE). You are free to modify and redistribute the code provided proper credit and copyright notices are maintained.
* **Pre-trained Model Weights (`k2-fsa/OmniVoice`):** Distributed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only.
