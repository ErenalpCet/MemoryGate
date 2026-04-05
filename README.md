# MemoryGate

> A lightweight DistilBERT classifier that decides what an AI assistant should remember — and what it should forget.

Most AI assistants treat every conversation turn equally. MemoryGate filters them by importance, so only meaningful information gets stored in long-term memory — medical details, deadlines, passwords, personal events — while casual small talk is quietly discarded. All memories are encrypted on disk; nothing leaves your machine.

> **Warning:** Do not use a reasoning model in LM Studio. Reasoning models may break the tool-calling system.

> Tested and verified on Ubuntu. Windows users see the [platform notes](#platform-notes) below.

<img width="1008" height="476" alt="MemoryGate in action" src="https://github.com/user-attachments/assets/3dbec8b9-d578-4d90-8fb0-524ce57a24a4" />

---

## How It Works

MemoryGate runs as a three-stage pipeline.

**Stage 1 — Generate:** A local LLM (via LM Studio) produces labelled training examples covering high- and low-importance conversation topics.

**Stage 2 — Train:** A DistilBERT classifier is fine-tuned on that data to score each conversation turn.

**Stage 3 — Run:** The trained model runs in real time. Each assistant reply is scored; only important turns are saved to the encrypted RAG memory store.

```
Build pipeline (one-time)
─────────────────────────────────────────────────────────────
  LM Studio  ──►  generate_training_data.py  ──►  train_model.py
                         (.jsonl)                  (importance_model/)

Runtime (every conversation)
─────────────────────────────────────────────────────────────
  User input
      │
      ▼
  LM Studio  ◄──►  Web search (DuckDuckGo)
      │
      ▼
  Response
      │
      ▼
  Importance scorer  ──► (if score ≥ threshold) ──► RAG memory
      │                                                  │
      └──────────── retrieved context ◄──────────────────┘
```

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

- Python 3.10 (Anaconda recommended)
- LM Studio running locally with a non-reasoning model loaded
- A CUDA-capable GPU is recommended for training; CPU fallback is supported for inference

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

This installs PyTorch with CUDA 12.6 support. If you are on CPU only, remove the `--extra-index-url` line from `requirements.txt` before running.

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and adjust the LM Studio URL if you changed the default port.

### 5. Set up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai).
2. Download any instruction-tuned model (e.g. Mistral 7B Instruct, LLaMA 3 8B Instruct). **Do not use a reasoning model** — these break the tool-calling pipeline.
3. Go to the **Local Server** tab in LM Studio and click **Start Server**. The default address is `http://localhost:1234`.
4. Load your chosen model into the server.

---

## Usage

### Step 1 — Generate training data

With LM Studio running and a model loaded:

```bash
python generate_training_data.py
```

This produces `conversation_data.jsonl` with balanced high- and low-importance examples. Generation targets 800 examples per class by default and resumes from where it left off if interrupted.

### Step 2 — Train the model

```bash
python train_model.py
```

The best checkpoint (by validation loss) is saved to `./importance_model/`. Training takes a few minutes on a GPU and up to an hour on CPU.

### Step 3 — Run the memory filter

```bash
python run_memory.py
```

On first launch you will be prompted to create a master password. This password encrypts all stored memories and your identity profile. **There is no password recovery — keep it safe.**

---

## Project Structure

```
MemoryGate/
├── generate_training_data.py   # Synthetic data generation via LM Studio
├── train_model.py              # DistilBERT fine-tuning pipeline
├── run_memory.py               # Runtime memory filtering and chat loop
├── conversation_data.jsonl     # Generated training data (git-ignored)
├── importance_model/           # Saved model weights (git-ignored)
├── rag_memory_db/              # ChromaDB vector store (git-ignored)
├── .env.example                # Environment variable template
└── requirements.txt
```

---

## Configuration

Key settings in `train_model.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `model_name` | `distilbert-base-uncased` | Base transformer model |
| `batch_size` | `32` | Reduce if you run out of VRAM |
| `epochs` | `6` | Training epochs |
| `importance_threshold` | `0.60` | Classification threshold at training time |
| `use_amp` | `True` | Mixed precision — recommended for CUDA |

The threshold baked into the saved `config.json` is used at inference time by `run_memory.py`.

---

## Platform Notes

**Ubuntu / Linux** — fully supported and tested.

**macOS** — should work but is untested. MPS (Apple Silicon GPU) is not explicitly configured; the code falls back to CPU.

**Windows** — the code is compatible but there are two known friction points:
- `sounddevice` may require a manual PortAudio installation.
- `kokoro` TTS may need additional native library dependencies. Voice features are optional — the text-based chat path works without them.

---

## Troubleshooting

**"Cannot reach LM Studio"** — Make sure the Local Server is running in LM Studio (not just the chat interface) and that a model is loaded into it. Check that `LM_STUDIO_BASE_URL` in your `.env` matches the address shown in LM Studio.

**"Model not found in ./importance_model"** — You need to run `train_model.py` before `run_memory.py`. The model directory is created during training.

**Generation is slow or produces empty batches** — LM Studio may be overloaded or using a model that is too large for your hardware. Try a smaller model, or reduce `BATCH_SIZE` in `generate_training_data.py`.

**Training runs out of memory** — Reduce `batch_size` in `train_model.py` (try 8 or 16). If you are on CPU, also set `use_amp` to `False`.

**Voice input is not transcribing correctly** — Whisper large-v3-turbo requires around 4 GB of VRAM. If your GPU is smaller, swap `WHISPER_MODEL` in `run_memory.py` to `openai/whisper-small` or `openai/whisper-base`.

---

## License

Licensed under the GNU Affero General Public License v3.0. Any project that uses MemoryGate — including over a network or API — must also be released under AGPL-3.0. See [LICENSE](LICENSE) for full details.

---

## Author

**ErenalpCet** — Erenalp Çetintürk
