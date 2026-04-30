# MemoryGate

> A lightweight DistilBERT classifier that decides what an AI assistant should remember — and what it should forget.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)](https://github.com/ErenalpCet/MemoryGate)

Most AI assistants treat every conversation turn equally. MemoryGate filters them by importance, so only meaningful information gets stored in long-term memory — medical details, deadlines, passwords, personal events — while casual small talk is quietly discarded. All memories are encrypted on disk; nothing leaves your machine.

> **Warning:** Do not use a reasoning model in LM Studio. Reasoning models may break the tool-calling pipeline.

> Tested and verified on Ubuntu. Windows users see the [platform notes](#platform-notes) below.

<img width="1008" height="476" alt="MemoryGate in action" src="https://github.com/user-attachments/assets/3dbec8b9-d578-4d90-8fb0-524ce57a24a4" />

---

## Table of Contents

- [How It Works](#how-it-works)
- [What Counts as Important](#what-counts-as-important)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Chat Commands](#chat-commands)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Platform Notes](#platform-notes)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## How It Works

MemoryGate runs as a three-stage pipeline.

**Stage 1 — Generate:** A local LLM (via LM Studio) produces labelled training examples covering high- and low-importance conversation topics.

**Stage 2 — Train:** A DistilBERT classifier is fine-tuned on that data to score each conversation turn.

**Stage 3 — Run:** The trained model runs in real time. Each assistant reply is scored; only important turns are saved to the encrypted RAG memory store.

```
Build pipeline (one-time)
────────────────────────────────────────────────────────────
  LM Studio  ──►  generate_training_data.py  ──►  train_model.py
                         (.jsonl)                  (importance_model/)

Runtime (every conversation)
────────────────────────────────────────────────────────────
  User input
      │
      ▼
  LM Studio  ◄──►  Web search (DuckDuckGo, no API key)
      │
      ▼
  Response
      │
      ▼
  Importance scorer  ──► (score ≥ threshold) ──► RAG memory (encrypted)
      │                                                 │
      └──────────── retrieved context ◄────────────────┘
```

### Key Design Decisions

- **Privacy-first:** The local LLM runs entirely on your machine via LM Studio. No data is sent to external AI APIs.
- **Zero-trust encryption:** All memories and your identity profile are encrypted with Fernet (AES-128-CBC) using a key derived from your master password via PBKDF2HMAC with 600,000 iterations.
- **Selective memory:** DistilBERT scores each turn so only genuinely important exchanges consume storage. Casual chitchat is discarded automatically.
- **Transparent web search:** The assistant calls DuckDuckGo automatically when it needs current information. Search results are displayed and can optionally be saved alongside the conversation.

---

## What Counts as Important

**High importance (label = 1)**
- Deaths, grief, family emergencies, personal trauma
- Passwords, API keys, PINs, access tokens
- Medical diagnoses, prescriptions, allergies, surgery dates
- Legal contracts, compliance deadlines, court dates
- Financial decisions, bank details, tax deadlines
- Project deadlines, stakeholder agreements, production credentials

**Low importance (label = 0)**
- Casual greetings and small talk
- General trivia and history facts
- Creative requests like jokes or poems
- Simple definitions and basic questions
- Movie or food recommendations

---

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.10 (Anaconda strongly recommended) |
| LM Studio | Running locally with a non-reasoning model loaded |
| GPU | CUDA-capable GPU recommended for training; CPU is supported for inference |
| Disk space | ~1 GB for DistilBERT weights + ChromaDB |
| RAM | 8 GB minimum; 16 GB recommended |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ErenalpCet/MemoryGate.git
cd MemoryGate
```

### 2. Create a Python 3.10 environment

```bash
conda create -n memorygate python=3.10
conda activate memorygate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch with CUDA 12.6 support by default. If you are on CPU only, remove or comment out the `--extra-index-url` line in `requirements.txt` before running.

> **Voice features are optional.** If you do not need voice input/output (e.g. on a headless server), you can skip installing `sounddevice` and `kokoro`. The text-based chat path works without them.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and adjust `LM_STUDIO_BASE_URL` if you changed the default LM Studio port. The defaults work out of the box for a standard LM Studio install.

### 5. Set up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai).
2. Download any instruction-tuned model. Recommended options:
   - **Mistral 7B Instruct** — good balance of speed and quality
   - **LLaMA 3 8B Instruct** — excellent tool-calling support
   - **Qwen 2.5 7B Instruct** — strong multilingual performance
3. **Do not use a reasoning model** (e.g. DeepSeek-R1, QwQ) — these produce non-standard output formats that break the tool-calling pipeline.
4. Go to the **Local Server** tab and click **Start Server**. The default address is `http://localhost:1234`.
5. Load your chosen model into the server.

---

## Quick Start

Once LM Studio is running with a model loaded:

```bash
# Step 1: Generate training data (~800 high + 800 low importance examples)
python generate_training_data.py

# Step 2: Train the importance classifier (a few minutes on GPU, up to 1 hour on CPU)
python train_model.py

# Step 3: Start the memory-filtered chat assistant
python run_memory.py
```

On first launch you will be prompted to create a master password. This password is the only key to your encrypted memories — **there is no recovery mechanism if you lose it.**

---

## Usage

### Step 1 — Generate training data

```bash
python generate_training_data.py
```

This produces `conversation_data.jsonl` with balanced high- and low-importance examples. Generation targets 800 examples per class by default and **resumes from where it left off** if interrupted, so you can safely stop and restart it.

### Step 2 — Train the model

```bash
python train_model.py
```

The best checkpoint (lowest validation loss) is saved to `./importance_model/`. You will see per-epoch metrics showing training loss, validation loss, and accuracy at both the 50% and configured thresholds.

### Step 3 — Run the memory filter

```bash
python run_memory.py
```

After entering your password, the chat loop starts. Each assistant reply is automatically scored. If the score meets the threshold, the exchange is encrypted and saved to the RAG store. Future queries retrieve semantically similar memories and inject them as context.

---

## Chat Commands

| Command | Description |
|---|---|
| `/save` | Manually save the last conversation turn (useful when the auto-scorer disagrees). If the last turn included a web search, the query and results are saved alongside it. |
| `/memories` | List all saved memories with their importance scores and flags (`[M]` = manually saved, `[S]` = includes search results). |
| `/search <query>` | Semantic search over saved memories. |
| `/clear` | Permanently delete all saved memories. |
| `/identity <text>` | Set your identity/preferences (name, job, health conditions, communication style, etc.). This text is injected into every system prompt. |
| `/identity view` | Display your current identity. |
| `/identity edit` | Edit your existing identity interactively. |
| `/identity clear` | Delete your identity. |
| `/voice` | Toggle voice input on/off. You will be prompted to select a microphone on first enable. |
| `/exit` or `/quit` | Exit the program. |

**Web search is fully automatic.** Just ask about current events, prices, recent documentation, or anything the local model would not know. The assistant decides when to search. After a search-backed response, use `/save` to persist the results to memory.

---

## Configuration

### `train_model.py`

| Setting | Default | Description |
|---|---|---|
| `model_name` | `distilbert-base-uncased` | Base transformer model for the classifier |
| `batch_size` | `32` | Reduce to 8 or 16 if you run out of VRAM |
| `epochs` | `6` | Training epochs |
| `lr` | `2e-5` | Learning rate |
| `importance_threshold` | `0.60` | Minimum score for a turn to be saved automatically |
| `use_amp` | `True` | Mixed precision training — recommended for CUDA. Set to `False` for CPU. |
| `dropout` | `0.3` | Dropout rate in the classifier head |

The threshold baked into `importance_model/config.json` is read at inference time by `run_memory.py`. To change the threshold without retraining, edit it directly in that file.

### `run_memory.py`

| Setting | Default | Description |
|---|---|---|
| `MAX_HISTORY_TURNS` | `20` | Number of conversation turns kept in the context window. Reduce for models with small context windows (e.g. 4k tokens). |
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | Speech-to-text model. Swap to `openai/whisper-small` if VRAM is limited. |
| `VAD_SILENCE_SECS` | `5.0` | Seconds of silence before voice recording stops. |
| `KOKORO_VOICE` | `af_heart` | TTS voice for Kokoro. |
| `EMBEDDER_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer used for RAG retrieval embeddings. |

### `generate_training_data.py`

| Setting | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `8` | Examples requested per LLM call. Reduce if LM Studio times out. |
| `TARGET_HIGH` | `800` | Target number of high-importance examples. |
| `TARGET_LOW` | `800` | Target number of low-importance examples. |
| `TEMPERATURE` | `0.85` | Generation temperature. Higher = more varied examples. |

---

## Project Structure

```
MemoryGate/
├── generate_training_data.py   # Synthetic data generation via LM Studio
├── train_model.py              # DistilBERT fine-tuning pipeline
├── run_memory.py               # Runtime memory filtering and chat loop
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── conversation_data.jsonl     # Generated training data (git-ignored)
├── importance_model/           # Saved model weights and config (git-ignored)
│   ├── best_model.pt
│   ├── config.json
│   └── tokenizer files
├── rag_memory_db/              # ChromaDB vector store (git-ignored)
├── user_identity.enc           # Encrypted identity profile (git-ignored)
├── rag_memory.salt             # Encryption salt (git-ignored, do not delete)
└── .setup_token                # Password verification token (git-ignored)
```

> **Do not delete `rag_memory.salt`.** This file is required to derive the encryption key from your password. Losing it makes all stored memories permanently unreadable.

---

## Platform Notes

**Ubuntu / Linux** — fully supported and tested.

**macOS** — should work but is untested. MPS (Apple Silicon GPU) is not explicitly configured; the code falls back to CPU. If you encounter issues with `sounddevice`, install PortAudio via Homebrew: `brew install portaudio`.

**Windows** — the code is compatible but there are two known friction points:
- `sounddevice` requires a manual PortAudio installation. Download the appropriate binary from the [PortAudio website](http://www.portaudio.com) or install via `conda install portaudio`.
- `kokoro` TTS may need additional native library dependencies. Voice features are entirely optional — the text-based chat path works without them and without installing either `sounddevice` or `kokoro`.

---

## Troubleshooting

**"Cannot reach LM Studio"**
Make sure the Local Server tab is active in LM Studio (not just the chat interface) and that a model is fully loaded. Check that `LM_STUDIO_BASE_URL` in `.env` matches the address shown in LM Studio.

**"Model not found in ./importance_model"**
You must run `train_model.py` before `run_memory.py`. The directory is created during training.

**Generation is slow or produces empty batches**
LM Studio may be overloaded or the loaded model is too large for your hardware. Try a smaller model (3B–7B parameters), or reduce `BATCH_SIZE` in `generate_training_data.py` to 4.

**Training runs out of memory**
Reduce `batch_size` in `train_model.py` (try 8 or 16). On CPU, also set `use_amp` to `False`.

**Voice input is not transcribing correctly**
`openai/whisper-large-v3-turbo` requires around 4 GB of VRAM. If your GPU is smaller, swap `WHISPER_MODEL` in `run_memory.py` to `openai/whisper-small` or `openai/whisper-base`.

**Memories from a previous session are unreadable / decryption fails**
This happens if `rag_memory.salt` was deleted or the wrong password was entered. The salt and password must both match what was used when the memories were saved. There is no bypass.

**Web search returns no results**
DuckDuckGo rate-limits aggressive queries. Wait a moment and try again, or rephrase the query. If the error persists, check that `ddgs` is installed: `pip install ddgs`.

**`/save` after a search does not include search results**
Only the most recent search is eligible for manual saving. If you had a search turn followed by a non-search turn before running `/save`, the search data will have been cleared. Always run `/save` immediately after a search-backed response.

---

## Known Limitations

- **No password recovery.** If you forget your master password, all encrypted memories and your identity profile are permanently inaccessible.
- **Context window cap.** `MAX_HISTORY_TURNS` (default 20) limits how much conversation history is sent to LM Studio. Very long sessions may lose early context.
- **LLM tool-calling reliability.** Whether the local model chooses to call `web_search` depends on the model's instruction-following quality. Smaller or poorly fine-tuned models may ignore the tool even when it would be helpful.
- **Single-user only.** All memories, encryption keys, and identity data are tied to a single master password on the local machine. There is no multi-user support.

---

## License

Licensed under the GNU Affero General Public License v3.0. Any project that uses or distributes MemoryGate — including over a network or API — must also be released under AGPL-3.0. See [LICENSE](LICENSE) for full details.

---

## Author

**ErenalpCet** — Erenalp Çetintürk
