# MemoryGate

A lightweight DistilBERT classifier that decides what an AI assistant should remember — and what it should forget.

Most AI assistants treat all conversation turns equally. MemoryGate filters them by importance, so only meaningful information gets stored in long-term memory — things like medical details, deadlines, passwords, and personal events — while casual small talk and trivia are quietly discarded.

---

## How It Works

MemoryGate is a three-stage pipeline:

1. **Generate** — Uses a local LLM via LM Studio to produce labelled training examples across high and low importance conversation topics
2. **Train** — Fine-tunes a DistilBERT classifier on that data to score each conversation turn
3. **Run** — Applies the trained model in real time to decide what the assistant should commit to memory

---

## What Counts as Important

High importance (label = 1):
- Deaths, grief, family emergencies, personal trauma
- Passwords, API keys, PINs, access tokens
- Medical diagnoses, prescriptions, allergies, surgery dates
- Legal contracts, compliance deadlines, court dates
- Financial decisions, bank details, tax deadlines
- Project deadlines, stakeholder agreements, production credentials

Low importance (label = 0):
- Casual greetings and small talk
- General trivia and history facts
- Creative requests like jokes or poems
- Simple definitions and basic questions
- Movie or food recommendations

---

## Requirements

- Python 3.9 or higher
- A CUDA-capable GPU is recommended for training (CPU fallback is supported)
- LM Studio running locally with a model loaded (only needed for data generation)

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/MemoryGate.git
cd MemoryGate
```

Install dependencies:

```
pip install -r requirements.txt
```

This will automatically install PyTorch with CUDA 12.6 support. If you are on CPU only, replace the `--index-url` line in `requirements.txt` with the standard PyPI version.

Set up your environment variables by copying the example file:

```
cp .env.example .env
```

Then open `.env` and adjust the settings if needed.

---

## Usage

### Step 1 — Generate Training Data

Make sure LM Studio is running with a model loaded, then run:

```
python generate_training_data.py
```

This produces `conversation_data.jsonl` with balanced high and low importance examples.

### Step 2 — Train the Model

```
python train_model.py
```

The best checkpoint is saved to `./importance_model/` based on validation loss.

### Step 3 — Run the Memory Filter

```
python run_memory.py
```

---

## Project Structure

```
MemoryGate/
├── generate_training_data.py   # Synthetic data generation via LM Studio
├── train_model.py              # DistilBERT fine-tuning pipeline
├── run_memory.py               # Runtime memory filtering
├── conversation_data.jsonl     # Generated training data (git ignored)
├── importance_model/           # Saved model weights (git ignored)
├── .env.example                # Environment variable template
└── requirements.txt
```

---

## Configuration

Key settings in `train_model.py`:

- `model_name` — base model, default is distilbert-base-uncased
- `batch_size` — adjust based on your available VRAM
- `epochs` — default is 6
- `importance_threshold` — deployment threshold, default is 0.60
- `use_amp` — mixed precision training, recommended for CUDA

---

## License

This project is licensed under the GNU Affero General Public License v3.0.

Any project that uses MemoryGate — including over a network or API — must also be released under AGPL-3.0. See the LICENSE file for full details.

---

## Author

ErenalpCet / Erenalp Çetintürk
