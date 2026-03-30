import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]✅ CUDA:[/green] {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠️ CPU fallback[/yellow]")
    return device

DEVICE = get_device()

CFG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "batch_size": 32,        # Adjust based on your available VRAM
    "epochs": 6,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "save_dir": "./importance_model",
    "seed": 42,
    "importance_threshold": 0.60,
    "use_amp": True,
    "amp_dtype_str": "float16",
    "use_compile": True,
}

_AMP_DTYPE = (
    torch.float16 if CFG["amp_dtype_str"] == "float16"
    else torch.bfloat16 if CFG["amp_dtype_str"] == "bfloat16"
    else torch.float16
)

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

GENERATED_DATA_PATH = "./conversation_data.jsonl"

if not os.path.exists(GENERATED_DATA_PATH):
    console.print(f"[red]❌ No training data at {GENERATED_DATA_PATH}. Run generate_training_data.py first.[/red]")
    raise SystemExit(1)

with open(GENERATED_DATA_PATH, encoding="utf-8") as f:
    SYNTHETIC_DATA = [json.loads(line.strip()) for line in f if line.strip()]

console.print(f"[blue]📂 Loaded {len(SYNTHETIC_DATA)} examples[/blue]")

class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"USER: {item['input']} ASSISTANT: {item['output']}"
        enc = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.float),
        }

class ImportanceClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_vec).squeeze(-1)

def train():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    train_data, val_data = train_test_split(SYNTHETIC_DATA, test_size=0.2, random_state=CFG["seed"], stratify=[d["label"] for d in SYNTHETIC_DATA])
    train_ds = ConversationDataset(train_data, tokenizer, CFG["max_length"])
    val_ds = ConversationDataset(val_data, tokenizer, CFG["max_length"])

    _on_gpu = torch.cuda.is_available()
    _num_workers = min(8, os.cpu_count() or 1) if _on_gpu else 0
    loader_kwargs: dict = {
        "batch_size": CFG["batch_size"],
        "num_workers": _num_workers,
        "pin_memory": _on_gpu,
        "persistent_workers": _on_gpu and _num_workers > 0,
    }
    if _num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = ImportanceClassifier(CFG["model_name"], CFG["dropout"]).to(DEVICE)
    if CFG.get("use_compile") and torch.cuda.is_available() and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss()

    use_amp = CFG.get("use_amp") and torch.cuda.is_available()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_val_loss = float("inf")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
                    logits = model(ids, mask)
                    loss = criterion(logits, label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, mask)
                loss = criterion(logits, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = correct_t = total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                label = batch["label"].to(DEVICE, non_blocking=True)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
                        logits = model(ids, mask)
                        val_loss += criterion(logits, label).item()
                else:
                    logits = model(ids, mask)
                    val_loss += criterion(logits, label).item()

                preds_50 = (logits >= 0.0).float()
                thresh_logit = torch.log(
                    torch.tensor(CFG["importance_threshold"] / (1.0 - CFG["importance_threshold"]))
                ).to(DEVICE)
                preds_thresh = (logits >= thresh_logit).float()

                correct    += (preds_50     == label).sum().item()
                correct_t  += (preds_thresh == label).sum().item()
                total += label.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct   / total
        val_acc_t = correct_t / total

        console.print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc@50%={val_acc:.2%} val_acc@{CFG['importance_threshold']:.0%}={val_acc_t:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), os.path.join(CFG["save_dir"], "best_model.pt"))

    tokenizer.save_pretrained(CFG["save_dir"])
    save_cfg = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in CFG.items()}
    with open(os.path.join(CFG["save_dir"], "config.json"), "w") as f:
        json.dump(save_cfg, f, indent=2, default=str)

    console.print(Panel.fit(
        f"[green]Training complete. Best val_loss = {best_val_loss:.4f}[/green]\n"
        f"Model saved to: {CFG['save_dir']}/",
        border_style="green"
    ))

if __name__ == "__main__":
    train()
