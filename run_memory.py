from __future__ import annotations
import json
import os
import sys
import uuid
import re
from datetime import datetime, timezone
from collections import deque
from dotenv import load_dotenv
import chromadb
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, pipeline as hf_pipeline

load_dotenv()

# ====================== CONFIG ======================
LM_STUDIO_BASE_URL   = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY    = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
IMPORTANCE_MODEL_DIR = "./importance_model"
RAG_DB_PATH          = "./rag_memory_db"
EMBEDDER_MODEL       = "all-MiniLM-L6-v2"
IDENTITY_FILE        = "./user_identity.enc"
SALT_FILE            = "./rag_memory.salt"
SETUP_TOKEN_FILE     = "./.setup_token"
WHISPER_MODEL        = "openai/whisper-large-v3-turbo"
MIC_CHANNELS         = 1
KOKORO_VOICE         = "af_heart"
KOKORO_SPEED         = 1.0
KOKORO_SAMPLE_RATE   = 24000

# Voice Activity Detection
VAD_ENERGY_THRESHOLD = 0.08
VAD_SILENCE_SECS     = 5.0
VAD_MIN_SPEECH_SECS  = 0.3
VAD_PRE_ROLL_SECS    = 0.5
VAD_FRAME_SECS       = 0.03

# FIX: Maximum chat history turns to keep to avoid context window overflow.
# Each turn = 2 messages (user + assistant). 20 turns = 40 messages.
# Adjust down if your LM Studio model has a small context window (e.g. 4k).
MAX_HISTORY_TURNS = 20

SYSTEM_PROMPT = """You are a knowledgeable, concise assistant powered by MemoryGate.

=== SYSTEM DESIGN ===
MemoryGate was created by Erenalp Çetintürk / ErenalpCet.
This is a Zero-Trust Encrypted Memory System with:
- RAG (Retrieval-Augmented Generation) for contextual memory
- Importance Scoring (DistilBERT classifier) to save only meaningful conversations
- End-to-end encryption (Fernet + PBKDF2HMAC) for all stored data
- Voice I/O (Whisper STT + Kokoro TTS) for hands-free interaction
- Local LLM via LM Studio for privacy-first AI
- Web Search (DuckDuckGo, no API key) — call web_search for current info

=== YOUR BEHAVIOR ===
- Use relevant memories when provided to personalize responses
- Keep spoken responses clear and short (optimized for voice)
- Always respond in English unless explicitly told otherwise
- Be aware of the current date and time provided in each conversation
- When you need current or recent information or facts you are unsure about,
  call the web_search tool BEFORE answering — don't guess
=== END ===
"""

# ====================== WEB SEARCH TOOL DEFINITION ======================
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current information, news, prices, documentation, "
            "or any topic you are unsure about. Returns JSON with title, url, and snippet "
            "for each result. Use this whenever you need up-to-date or specific facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to look up online."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default: 5.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
console = Console(force_terminal=True)

# ====================== PRE-COMPILED REGEX (performance) ======================
_TTS_CLEANUP_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\[.*?\]"),                  ""),
    (re.compile(r"\*{1,3}[^\n*]*?\*{1,3}"),  ""),
    (re.compile(r"`{1,3}[^\n`]*?`{1,3}"),    ""),
    (re.compile(r"#{1,6}\s+"),               ""),
    (re.compile(r"^>\s+", re.MULTILINE),     ""),
]


# ====================== WEB SEARCHER ======================
class WebSearcher:
    def search(self, query: str, max_results: int = 5) -> dict:
        try:
            from ddgs import DDGS
        except ImportError:
            return {
                "query": query,
                "results": [],
                "error": "ddgs not installed. Run: pip install ddgs",
            }

        try:
            max_results = max(1, min(10, int(max_results)))
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))

            if not raw:
                return {"query": query, "results": [], "error": "No results found."}

            results = [
                {
                    "title":   r.get("title",   "").strip(),
                    "url":     r.get("href",    "").strip(),
                    "snippet": r.get("body",    "").strip(),
                }
                for r in raw
            ]
            return {"query": query, "results": results, "error": None}

        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc)}

    @staticmethod
    def format_for_display(data: dict) -> str:
        if data.get("error") and not data.get("results"):
            return f"[red]Search error: {data['error']}[/red]"
        lines = [f"[bold]Search:[/bold] {data['query']}"]
        for i, r in enumerate(data["results"], 1):
            lines.append(
                f"  [cyan]{i}.[/cyan] [bold]{r['title']}[/bold]\n"
                f"     [dim]{r['url']}[/dim]\n"
                f"     {r['snippet'][:200]}"
            )
        return "\n".join(lines)


# ====================== SECURE PASSWORD INPUT ======================
def secure_password(prompt_text: str = "Master password") -> str:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Prompt.ask(f"[bold yellow]{prompt_text}[/bold yellow]", password=True)


# ====================== PASSWORD SYSTEM ======================
class MemoryEncryptor:
    def __init__(self, password: str):
        self._fernet = self._derive_key(password)

    def _derive_key(self, password: str) -> Fernet:
        if not os.path.exists(SALT_FILE):
            salt = os.urandom(16)
            with open(SALT_FILE, "wb") as f:
                f.write(salt)
            console.print("[green]Encryption salt created[/green]")
        else:
            with open(SALT_FILE, "rb") as f:
                salt = f.read(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return Fernet(key)

    def encrypt(self, text: str) -> str:
        return self._fernet.encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode()).decode()


def first_time_setup():
    console.print(Panel.fit(
        "[bold magenta]FIRST TIME SETUP[/bold magenta]\n"
        "Please create your master password.\n"
        "This password will encrypt all memories and your identity.",
        border_style="magenta"
    ))
    while True:
        pwd1 = secure_password("Set master password")
        pwd2 = secure_password("Confirm master password")
        if pwd1 == pwd2 and pwd1:
            return pwd1
        console.print("[red]Passwords do not match. Please try again.[/red]")


def verify_password(password: str) -> bool:
    if not os.path.exists(SETUP_TOKEN_FILE):
        return False
    try:
        encryptor = MemoryEncryptor(password)
        with open(SETUP_TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
        encryptor.decrypt(token)
        return True
    except Exception:
        return False


def create_setup_token(password: str):
    encryptor = MemoryEncryptor(password)
    token = encryptor.encrypt("SETUP_OK")
    with open(SETUP_TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(token)


# ====================== IMPORTANCE SCORER ======================
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)


class ImportanceScorer:
    def __init__(self, model_dir: str = IMPORTANCE_MODEL_DIR):
        self.model_dir = model_dir
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        # FIX: removed hardcoded 0.65 default — threshold is always read from
        # config.json in _load(), so a stale default here was misleading.
        self.threshold: float = 0.60
        self._load()

    def _load(self):
        cfg_path   = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "best_model.pt")
        if not os.path.exists(cfg_path) or not os.path.exists(model_path):
            raise RuntimeError(f"Model not found in {self.model_dir}")

        with open(cfg_path) as f:
            cfg = json.load(f)

        # Threshold always comes from the saved config — never from a hardcoded default.
        self.threshold = cfg.get("importance_threshold", 0.60)
        model_name     = cfg.get("model_name", "distilbert-base-uncased")
        dropout        = cfg.get("dropout", 0.3)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model     = ImportanceClassifier(model_name, dropout)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

        if hasattr(torch, "compile") and torch.cuda.is_available():
            self.model = torch.compile(self.model)

        console.print(
            f"[green]Importance classifier loaded[/green] "
            f"threshold=[yellow]{self.threshold:.0%}[/yellow]"
        )

    @torch.no_grad()
    def score(
        self,
        user_input: str,
        assistant_output: str,
        chat_history: list | None = None,
    ) -> float:
        if chat_history and len(chat_history) >= 4:
            recent  = chat_history[-6:]
            context = "\n".join(
                f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}"
                for t in recent
            )
            text = f"{context}\nUSER: {user_input} ASSISTANT: {assistant_output}"
        else:
            text = f"USER: {user_input} ASSISTANT: {assistant_output}"

        enc  = self.tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(self.device, non_blocking=True)
        mask = enc["attention_mask"].to(self.device, non_blocking=True)

        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(ids, mask)
                score  = torch.sigmoid(logits).item()
        else:
            logits = self.model(ids, mask)
            score  = torch.sigmoid(logits).item()
        return round(score, 4)

    def is_important(self, score: float) -> bool:
        return score >= self.threshold


# ====================== RAG MEMORY ======================
class RAGMemory:
    COLLECTION = "important_conversations"

    def __init__(self, password: str):
        self.encryptor = MemoryEncryptor(password)
        self.embedder  = SentenceTransformer(
            EMBEDDER_MODEL,
            device=str(DEVICE),
            model_kwargs={
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
            },
        )
        self.client     = chromadb.PersistentClient(path=RAG_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        console.print(f"[green]RAG memory ready[/green] ({self.collection.count()} memories)")

    def save(
        self,
        user_input: str,
        assistant_output: str,
        score: float,
        manual: bool = False,
        search_data: dict | None = None,
    ) -> str:
        mem_id = str(uuid.uuid4())

        document_parts = [f"USER: {user_input}", f"ASSISTANT: {assistant_output}"]
        if search_data and search_data.get("results"):
            snippets = " | ".join(
                r.get("snippet", "")[:200] for r in search_data["results"][:5]
            )
            document_parts.append(f"SEARCH_CONTEXT: {snippets}")
        document  = "\n".join(document_parts)
        embedding = self.embedder.encode(document, normalize_embeddings=True).tolist()

        metadata: dict = {
            "user_input":       self.encryptor.encrypt(user_input[:600]),
            "assistant_output": self.encryptor.encrypt(assistant_output[:600]),
            "importance_score": score,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "manual_save":      manual,
            "has_search":       False,
        }

        if search_data:
            raw_json = json.dumps(search_data, ensure_ascii=False)[:3000]
            metadata["search_data"]  = self.encryptor.encrypt(raw_json)
            metadata["has_search"]   = True
            metadata["search_query"] = search_data.get("query", "")[:120]

        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[self.encryptor.encrypt(document)],
            metadatas=[metadata],
        )
        return mem_id

    def _decrypt_row(self, meta: dict) -> dict:
        try:
            out = {
                **meta,
                "user_input":       self.encryptor.decrypt(meta["user_input"]),
                "assistant_output": self.encryptor.decrypt(meta["assistant_output"]),
            }
        except Exception:
            out = {**meta, "user_input": "[DECRYPT FAILED]", "assistant_output": "[DECRYPT FAILED]"}

        if meta.get("search_data"):
            try:
                out["search_data"] = json.loads(self.encryptor.decrypt(meta["search_data"]))
            except Exception:
                out["search_data"] = None
        else:
            out["search_data"] = None

        return out

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.25,
    ) -> list[dict]:
        if self.collection.count() == 0:
            return []
        q_emb   = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self.collection.count()),
        )
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = round(1.0 - dist, 4)
            if sim >= min_similarity:
                plain = self._decrypt_row(meta)
                memories.append({
                    "similarity":       sim,
                    "importance_score": plain.get("importance_score", 0),
                    "timestamp":        plain.get("timestamp", ""),
                    "manual_save":      plain.get("manual_save", False),
                    "has_search":       plain.get("has_search", False),
                    "search_query":     plain.get("search_query", ""),
                    "search_data":      plain.get("search_data"),
                    "user_input":       plain["user_input"],
                    "assistant_output": plain["assistant_output"],
                })
        return sorted(memories, key=lambda x: x["similarity"], reverse=True)

    def list_all(self, limit: int = 50) -> list[dict]:
        if self.collection.count() == 0:
            return []
        r = self.collection.get(limit=limit)
        return [self._decrypt_row(m) for m in r["metadatas"]]

    def clear(self) -> int:
        n = self.collection.count()
        if n:
            self.collection.delete(ids=self.collection.get()["ids"])
        return n

    def count(self) -> int:
        return self.collection.count()


# ====================== IDENTITY ======================
def load_identity(encryptor: MemoryEncryptor) -> str:
    if not os.path.exists(IDENTITY_FILE):
        return ""
    try:
        with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
        return encryptor.decrypt(token)
    except Exception:
        return ""


def save_identity(text: str, encryptor: MemoryEncryptor):
    with open(IDENTITY_FILE, "w", encoding="utf-8") as f:
        f.write(encryptor.encrypt(text.strip()))


# ====================== COMMAND HELP ======================
def show_commands():
    console.print(Panel.fit(
        "[bold cyan]Available Commands[/bold cyan]\n\n"
        "[green]/identity[/green] <text>  -> Set your identity/preferences\n"
        "[green]/identity[/green] view   -> View current identity\n"
        "[green]/identity[/green] edit   -> Edit existing identity\n"
        "[green]/identity[/green] clear  -> Delete identity\n\n"
        "[green]/memories[/green]        -> List all saved memories\n"
        "[green]/search[/green] <query>  -> Search memories\n"
        "[green]/clear[/green]           -> Delete all memories\n\n"
        "[bold green]/save[/bold green]            -> Manually save last conversation\n"
        "                   [dim]If the last turn included a web search, the\n"
        "                   query + JSON results are saved alongside it.[/dim]\n\n"
        "[green]/voice[/green]           -> Toggle voice input\n"
        "[green]/exit[/green] or [green]/quit[/green] -> Exit program\n\n"
        "[dim]Web search is automatic — just ask about current events,\n"
        "   prices, docs, or anything the LLM should look up online.\n"
        "   After a search, use /save to persist the results.[/dim]",
        border_style="cyan",
        title="Commands"
    ))


# ====================== DATE/TIME HELPER ======================
def get_current_datetime() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return now.strftime("%A, %B %d, %Y - %I:%M %p %Z")


# ====================== VOICE ======================
class VoiceInput:
    def __init__(self):
        self.device_id  = None
        self.sample_rate = 16000
        self._whisper   = None

    def pick_microphone(self):
        devices    = sd.query_devices()
        input_devs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if not input_devs:
            console.print("[red]No microphone found.[/red]")
            return
        for i, (_, dev) in enumerate(input_devs):
            console.print(f"{i}: {dev['name']}")
        idx              = int(Prompt.ask("[bold yellow]Choose microphone number[/bold yellow]", default="0"))
        self.device_id   = input_devs[idx][0]
        self.sample_rate = int(sd.query_devices()[self.device_id]["default_samplerate"])
        console.print("[green]Microphone selected[/green]")
        console.print("[dim]Loading Whisper model ...[/dim]")
        self._load_whisper()
        console.print("[green]Whisper ready[/green]")

    def _load_whisper(self):
        if self._whisper is not None:
            return
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._whisper = hf_pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL,
                device=DEVICE,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                generate_kwargs={"language": "en", "task": "transcribe"},
            )

    def listen(self) -> str:
        if self.device_id is None:
            return ""
        self._load_whisper()
        console.print("[dim]Listening ...[/dim]")

        frame_samples        = int(self.sample_rate * VAD_FRAME_SECS)
        pre_roll_frames      = int(VAD_PRE_ROLL_SECS / VAD_FRAME_SECS)
        silence_frames_needed = int(VAD_SILENCE_SECS / VAD_FRAME_SECS)
        min_speech_frames    = int(VAD_MIN_SPEECH_SECS / VAD_FRAME_SECS)

        ring          = deque(maxlen=pre_roll_frames)
        speech_frames: list[np.ndarray] = []
        speaking      = False
        silence_count = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=MIC_CHANNELS,
            dtype="float32",
            device=self.device_id,
        ) as stream:
            while True:
                frame, _  = stream.read(frame_samples)
                frame_1d  = frame[:, 0] if frame.ndim > 1 else frame
                energy    = float(np.sqrt(np.mean(frame_1d ** 2)))

                if not speaking:
                    ring.append(frame_1d.copy())
                    if energy >= VAD_ENERGY_THRESHOLD:
                        speaking = True
                        speech_frames.extend(list(ring))
                        ring.clear()
                        silence_count = 0
                        console.print("[dim]Recording ...[/dim]")
                else:
                    speech_frames.append(frame_1d.copy())
                    if energy < VAD_ENERGY_THRESHOLD:
                        silence_count += 1
                        if silence_count >= silence_frames_needed:
                            break
                    else:
                        silence_count = 0

        if len(speech_frames) < min_speech_frames:
            return ""

        audio  = np.concatenate(speech_frames).astype(np.float32)
        result = self._whisper(
            {"sampling_rate": self.sample_rate, "raw": audio},
            return_timestamps=False,
        )
        text = result.get("text", "").strip()
        if text:
            console.print(f"[bold green]You[/bold green] (voice): {text}")
        return text


# ====================== LM STUDIO CLIENT ======================
class LMStudioClient:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
        self.model: str | None = None
        self._connect()

    def _connect(self):
        try:
            models = self.client.models.list()
            if models.data:
                self.model = models.data[0].id
                console.print(f"[green]LM Studio connected[/green] model=[cyan]{self.model}[/cyan]")
            else:
                console.print("[yellow]⚠ LM Studio is running but no model is loaded.[/yellow]")
                console.print("[dim]Load a model in LM Studio before chatting.[/dim]")
        except Exception as exc:
            console.print(f"[red]⚠ Cannot reach LM Studio at {LM_STUDIO_BASE_URL}[/red]")
            console.print(f"[dim]{exc}[/dim]")
            console.print("[dim]Start LM Studio, load a model, and make sure the server is on.[/dim]")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        if not self.model:
            raise RuntimeError("No model loaded in LM Studio.")
        full_reply = ""
        console.print("\n[bold cyan]Assistant[/bold cyan]")
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        return full_reply

    def chat_with_tools(
        self,
        messages: list[dict],
        searcher: WebSearcher,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        voice_notify: "callable | None" = None,
    ) -> tuple[str, dict | None]:
        if not self.model:
            raise RuntimeError("No model loaded in LM Studio.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        choice      = response.choices[0]
        search_data: dict | None = None

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = choice.message.tool_calls

            # FIX: added "content": None — the OpenAI spec requires this field on
            # assistant messages that contain tool_calls; its absence caused API
            # errors with some LM Studio backends.
            assistant_tool_msg: dict = {
                "role":       "assistant",
                "content":    None,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": tc.type,
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }

            tool_result_msgs: list[dict] = []
            for tc in tool_calls:
                if tc.function.name == "web_search":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    query       = args.get("query", "")
                    max_results = int(args.get("max_results", 5))

                    console.print(f"[dim]Web search -> [cyan]{query}[/cyan][/dim]")
                    if voice_notify:
                        voice_notify(f"Searching the web for: {query}")

                    search_data = searcher.search(query, max_results)

                    n = len(search_data.get("results", []))
                    if search_data.get("error") and not n:
                        console.print(f"[yellow]Search error: {search_data['error']}[/yellow]")
                        if voice_notify:
                            voice_notify(f"Search failed: {search_data['error']}")
                    else:
                        console.print(f"[dim]{n} result(s) found[/dim]")
                        if voice_notify:
                            voice_notify(f"Found {n} result{'s' if n != 1 else ''}. Preparing answer.")

                    tool_result_msgs.append({
                        "role":         "tool",
                        "content":      json.dumps(search_data, ensure_ascii=False),
                        "tool_call_id": tc.id,
                    })

            updated_messages = messages + [assistant_tool_msg] + tool_result_msgs
            full_reply = ""
            console.print("\n[bold cyan]Assistant[/bold cyan]")
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=updated_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply += delta
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return full_reply, search_data

        direct_content = choice.message.content or ""
        if direct_content:
            console.print("\n[bold cyan]Assistant[/bold cyan]")
            sys.stdout.write(direct_content + "\n")
            sys.stdout.flush()
            return direct_content, None

        return self.chat(messages, temperature, max_tokens), None


# ====================== VOICE SPEAKER ======================
class VoiceSpeaker:
    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError:
            console.print("[yellow]Kokoro not installed — TTS disabled. Run: pip install kokoro[/yellow]")
            return
        for kwargs in ({"lang_code": "a", "repo_id": "hexgrad/Kokoro-82M"}, {"lang_code": "a"}):
            try:
                self._pipeline = KPipeline(**kwargs)
                console.print("[green]Kokoro TTS ready[/green]")
                return
            except TypeError:
                continue
            except Exception as e:
                console.print(f"[yellow]Kokoro init error: {e}[/yellow]")
                return
        console.print("[yellow]Could not initialise Kokoro pipeline — TTS disabled.[/yellow]")

    def speak(self, text: str):
        self._load()
        if self._pipeline is None:
            return

        clean = text
        for pattern, repl in _TTS_CLEANUP_RE:
            clean = pattern.sub(repl, clean)
        clean = clean.replace("\u2013", "-").replace("\u2014", "-")
        clean = clean.replace("\u2018", "'").replace("\u2019", "'")
        clean = clean.replace("\u201c", '"').replace("\u201d", '"')

        if not clean:
            return

        chunks: list[np.ndarray] = []
        try:
            for result in self._pipeline(clean, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
                if hasattr(result, "audio"):
                    audio_chunk = result.audio
                elif isinstance(result, (tuple, list)) and len(result) >= 3:
                    audio_chunk = result[2]
                elif isinstance(result, (tuple, list)):
                    audio_chunk = result[-1]
                else:
                    audio_chunk = result

                if audio_chunk is None:
                    continue

                try:
                    import torch as _torch
                    if isinstance(audio_chunk, _torch.Tensor):
                        audio_chunk = audio_chunk.detach().cpu().numpy()
                except Exception:
                    pass

                try:
                    arr = np.asarray(audio_chunk, dtype=np.float32)
                    if arr.ndim == 0 or arr.size == 0:
                        continue
                    chunks.append(arr.ravel())
                except (ValueError, TypeError) as e:
                    console.print(f"[dim]TTS: skipping chunk — {e}[/dim]")
                    continue
        except Exception as e:
            console.print(f"[yellow]Kokoro synthesis error: {e}[/yellow]")
            return

        if not chunks:
            console.print("[yellow]TTS: Kokoro returned no audio chunks[/yellow]")
            return

        try:
            audio = np.concatenate(chunks).astype(np.float32)
            peak  = float(np.abs(audio).max())
            if peak == 0.0:
                console.print("[yellow]TTS: audio is silent, skipping playback[/yellow]")
                return
            if peak > 1.0:
                audio /= peak
            sd.play(audio, samplerate=KOKORO_SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            console.print(f"[yellow]Audio playback error: {e}[/yellow]")


# ====================== MAIN CHAT LOOP ======================
def run_chat():
    console.print(Panel.fit(
        "[bold cyan]MemoryGate — LM Studio + RAG Memory + Web Search + Voice I/O[/bold cyan]\n"
        "Zero-Trust Encrypted Memory System",
        border_style="cyan"
    ))

    if not os.path.exists(SETUP_TOKEN_FILE):
        password = first_time_setup()
        create_setup_token(password)
        console.print("[green]Master password set successfully![/green]")
    else:
        while True:
            try:
                password = secure_password()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Login cancelled.[/yellow]")
                sys.exit(0)
            if verify_password(password):
                console.print("[green]Password accepted[/green]")
                break
            console.print("[red]Wrong password. Try again, or press Ctrl-C to exit.[/red]")

    encryptor = MemoryEncryptor(password)
    scorer    = ImportanceScorer()
    memory    = RAGMemory(password)
    llm       = LMStudioClient()
    searcher  = WebSearcher()
    speaker   = VoiceSpeaker()
    mic       = VoiceInput()

    current_datetime = get_current_datetime()

    console.print(Panel.fit(
        f"[bold green]Welcome![/bold green]\n"
        f"Memories stored: [cyan]{memory.count()}[/cyan] | "
        f"Importance threshold: [yellow]{scorer.threshold:.0%}[/yellow]\n"
        f"[dim]{current_datetime}[/dim]",
        border_style="green"
    ))

    identity_text = load_identity(encryptor)
    if identity_text:
        console.print(f"[dim]Identity loaded ({len(identity_text)} chars)[/dim]")

    show_commands()

    voice_mode = Prompt.ask(
        "[bold yellow]Enable voice input? (y/n)[/bold yellow]", default="n"
    ).lower() in ("y", "yes")
    if voice_mode:
        mic.pick_microphone()

    chat_history: list[dict]  = []
    last_search_data: dict | None = None

    while True:
        user_input: str = ""

        if voice_mode:
            # FIX: catch mic/audio errors without crashing the session.
            # The user's pending text input is NOT lost — we fall back to a
            # text prompt in the same iteration instead of skipping via continue.
            try:
                user_input = mic.listen()
            except Exception as exc:
                console.print(f"[yellow]Voice input error: {exc}[/yellow]")
                console.print(
                    "[dim]Voice mode disabled. Use /voice to re-enable. "
                    "Enter your message as text below.[/dim]"
                )
                voice_mode = False
                # Fall through to text input below instead of continue-ing.

        if not voice_mode and not user_input:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                user_input = "/quit"

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()

            if cmd in ("/quit", "/exit"):
                console.print("[green]Goodbye![/green]")
                break

            elif cmd == "/voice":
                voice_mode = not voice_mode
                console.print(f"[green]Voice mode {'ON' if voice_mode else 'OFF'}[/green]")
                if voice_mode:
                    mic.pick_microphone()

            elif cmd == "/save":
                if chat_history and len(chat_history) >= 2:
                    last_user      = chat_history[-2]["content"]
                    last_assistant = chat_history[-1]["content"]
                    score          = 0.85

                    memory.save(
                        last_user,
                        last_assistant,
                        score,
                        manual=True,
                        search_data=last_search_data,
                    )

                    if last_search_data:
                        n_results = len(last_search_data.get("results", []))
                        console.print(
                            f"[green]Conversation + search saved![/green] "
                            f"([cyan]{n_results} web results[/cyan] for "
                            f"\"[italic]{last_search_data.get('query', '')}[/italic]\") "
                            f"Total memories: {memory.count()}"
                        )
                        last_search_data = None
                    else:
                        console.print(
                            f"[green]Conversation saved![/green] "
                            f"Total: {memory.count()}"
                        )
                else:
                    console.print("[yellow]No conversation to save yet.[/yellow]")

            elif cmd == "/memories":
                rows = memory.list_all()
                if rows:
                    t = Table(title="Saved Memories")
                    t.add_column("#")
                    t.add_column("Score")
                    t.add_column("Flags")
                    t.add_column("User")
                    t.add_column("Assistant")
                    for i, m in enumerate(rows, 1):
                        flags = ""
                        if m.get("manual_save"):
                            flags += "[M]"
                        if m.get("has_search"):
                            flags += "[S]"
                        t.add_row(
                            str(i),
                            f"{m.get('importance_score', 0):.1%}",
                            flags,
                            m["user_input"][:55],
                            m["assistant_output"][:55],
                        )
                    console.print(t)
                else:
                    console.print("[yellow]No memories yet.[/yellow]")

            elif cmd == "/search":
                query = user_input[8:].strip()
                if query:
                    results = memory.retrieve(query)
                    if results:
                        t = Table(title=f'Memory search: "{query}"')
                        t.add_column("Sim")
                        t.add_column("Score")
                        t.add_column("Flags")
                        t.add_column("User")
                        t.add_column("Assistant")
                        for m in results:
                            flags = ""
                            if m.get("manual_save"):
                                flags += "[M]"
                            if m.get("has_search"):
                                flags += "[S]"
                            t.add_row(
                                f"{m['similarity']:.1%}",
                                f"{m['importance_score']:.1%}",
                                flags,
                                m["user_input"][:55],
                                m["assistant_output"][:55],
                            )
                        console.print(t)
                    else:
                        console.print("[yellow]No matches.[/yellow]")

            elif cmd == "/clear":
                console.print(f"[red]Cleared {memory.clear()} memories.[/red]")

            elif cmd == "/identity":
                args = user_input[10:].strip()

                if args.lower() in ("clear", "delete", "reset"):
                    if os.path.exists(IDENTITY_FILE):
                        os.remove(IDENTITY_FILE)
                    console.print("[green]Identity cleared.[/green]")

                elif args.lower() == "view":
                    identity_text = load_identity(encryptor)
                    if identity_text:
                        console.print(Panel(
                            f"[white]{identity_text}[/white]",
                            title="Current Identity",
                            border_style="blue",
                        ))
                    else:
                        console.print("[yellow]No identity set.[/yellow]")

                elif args.lower() == "edit":
                    identity_text = load_identity(encryptor)
                    if identity_text:
                        console.print(f"[dim]Current identity ({len(identity_text)} chars):[/dim]")
                        console.print(f"[white]{identity_text}[/white]\n")
                        new_text = Prompt.ask(
                            "[bold yellow]Edit identity (press Enter to keep current)[/bold yellow]",
                            default=identity_text,
                        )
                        if new_text.strip():
                            save_identity(new_text, encryptor)
                            console.print("[green]Identity updated.[/green]")
                    else:
                        console.print("[yellow]No identity to edit. Use /identity <text> to set one.[/yellow]")

                elif args.lower() == "help":
                    console.print(Panel(
                        "[green]/identity[/green] <text>  -> Set new identity\n"
                        "[green]/identity[/green] view   -> View current\n"
                        "[green]/identity[/green] edit   -> Edit existing\n"
                        "[green]/identity[/green] clear  -> Delete",
                        title="Identity Commands",
                        border_style="yellow",
                    ))

                else:
                    if args:
                        save_identity(args, encryptor)
                        console.print("[green]Identity saved.[/green]")
                    else:
                        new_text = Prompt.ask("[bold yellow]Enter your identity/preferences[/bold yellow]")
                        if new_text.strip():
                            save_identity(new_text, encryptor)
                            console.print("[green]Identity saved.[/green]")

            continue

        # ── Normal conversation ──────────────────────────────────────────────
        relevant         = memory.retrieve(user_input, top_k=3)
        identity_text    = load_identity(encryptor)
        current_datetime = get_current_datetime()

        system_content  = SYSTEM_PROMPT
        system_content += f"\n\n=== CURRENT TIME ===\n{current_datetime}\n=== END ===\n"
        if identity_text:
            system_content += f"\n\n=== ABOUT YOU ===\n{identity_text}\n=== END ===\n"
        if relevant:
            mem_block = "\n\n".join(
                f"[Past - {m['importance_score']:.0%}]\nUser: {m['user_input']}\nAssistant: {m['assistant_output']}"
                for m in relevant
            )
            system_content += f"\n=== RELEVANT MEMORIES ===\n{mem_block}\n=== END ===\n"
            console.print(f"[dim]{len(relevant)} memories injected[/dim]")

        messages = [
            {"role": "system", "content": system_content},
            *chat_history,
            {"role": "user", "content": user_input},
        ]

        try:
            _notify = speaker.speak if voice_mode else None
            assistant_reply, search_data = llm.chat_with_tools(
                messages, searcher, voice_notify=_notify
            )
        except Exception as e:
            console.print(f"[red]LM Studio error: {e}[/red]")
            continue

        # FIX: always overwrite last_search_data (even with None) so that a
        # non-search turn following a search turn doesn't incorrectly attach
        # the old search results when the user runs /save.
        last_search_data = search_data

        if search_data:
            console.print(Panel(
                WebSearcher.format_for_display(search_data),
                title="Web Search Results (use /save to keep)",
                border_style="blue",
            ))

        if voice_mode:
            speaker.speak(assistant_reply)
            if search_data:
                speaker.speak("Say slash save to keep these web results in memory.")

        chat_history.extend([
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ])

        # FIX: trim history to MAX_HISTORY_TURNS pairs to avoid blowing the
        # model's context window. Pairs are trimmed from the oldest end.
        max_messages = MAX_HISTORY_TURNS * 2
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]

        score     = scorer.score(user_input, assistant_reply, chat_history)
        important = scorer.is_important(score)

        search_hint = (
            " [dim](includes web search — /save to persist results)[/dim]"
            if search_data else ""
        )
        if important:
            memory.save(user_input, assistant_reply, score, manual=False)
            console.print(Panel(
                f"Score: {score:.1%} SAVED | Total memories: {memory.count()}{search_hint}",
                title="Importance",
                border_style="green",
            ))
        else:
            console.print(Panel(
                f"Score: {score:.1%} not saved | Total memories: {memory.count()}\n"
                f"[dim]Use /save to manually save this conversation[/dim]{search_hint}",
                title="Importance",
                border_style="yellow",
            ))


if __name__ == "__main__":
    try:
        run_chat()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
