from __future__ import annotations
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

load_dotenv()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY  = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
OUTPUT_FILE        = os.getenv("OUTPUT_FILE", "./conversation_data.jsonl")

BATCH_SIZE  = 8
TARGET_HIGH = 800
TARGET_LOW  = 800
TEMPERATURE = 0.85
MAX_TOKENS  = 4096

console = Console(force_terminal=True)

HIGH_IMPORTANCE_TOPICS = [
    ("personal_tragedy",     1, "deaths in the family, grief, terminal illness, funerals, severe personal loss, emotional trauma, family emergencies"),
    ("security_credentials", 1, "passwords, pin codes, social security numbers, API keys, bank login details, secret codes, access tokens"),
    ("medical_critical",     1, "urgent medical situations, diagnoses, prescriptions, allergies, dosages, surgery dates, lab results"),
    ("legal_contracts",      1, "contract clauses, legal rulings, compliance deadlines, liability, regulatory requirements, court dates"),
    ("financial_decisions",  1, "budgets, revenue targets, investment amounts, bank account numbers, tax deadlines"),
    ("project_commitments",  1, "deadlines, stakeholder agreements, deliverable sign-offs, production credentials, server access"),
]

LOW_IMPORTANCE_TOPICS = [
    ("general_trivia",    0, "general knowledge, history facts, science trivia, geography, animal facts"),
    ("casual_chat",       0, "small talk, weather, jokes, 'how are you', 'how was your day', greetings"),
    ("creative_requests", 0, "writing poems, stories, brainstorming pet names, jokes"),
    ("simple_definitions",0, "asking for word meanings, basic math, how common appliances work"),
    ("recommendations",   0, "movie suggestions, food ideas, travel tips with no personal stakes"),
]

MULTI_TURN_HIGH = [
    ("serious_continuation", 1, "User continues talking about death, medical crisis, financial/legal issue — keep importance high"),
]

MULTI_TURN_LOW = [
    ("casual_after_serious", 0, "User says casual things like 'how are you', 'hi', 'thanks' AFTER a serious memory was injected — should stay LOW"),
]

SYSTEM_PROMPT = """You are a data generation assistant. Produce realistic labelled conversation examples.

LABEL DEFINITIONS (for the LAST turn only):
label=1 (IMPORTANT): The current exchange contains specific, actionable, or emotionally heavy info.
label=0 (NOT IMPORTANT): The current exchange is casual, general, or trivia.

OUTPUT FORMAT: Return ONLY a valid JSON array. No preamble.
Each element: {"input": "full recent context + current user message", "output": "assistant reply", "label": 0 or 1}"""

def _build_user_prompt(category: str, label: int, description: str, count: int) -> str:
    return (
        f"Generate exactly {count} examples for '{category}'. "
        f"Focus on: {description}. Label the LAST turn only. Return JSON array."
    )

class GeneratorClient:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
        self.model  = self._detect_model()

    def _detect_model(self) -> str:
        models = self.client.models.list()
        if not models.data:
            console.print("[red]❌ LM Studio running but no model loaded.[/red]")
            sys.exit(1)
        model_id = models.data[0].id
        console.print(f"[green]✅ LM Studio connected[/green] model=[cyan]{model_id}[/cyan]")
        return model_id

    def generate_batch(self, category: str, label: int, description: str, count: int) -> list[dict]:
        user_prompt = _build_user_prompt(category, label, description, count)
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        for attempt in range(1, 4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw = resp.choices[0].message.content.strip()
                return self._parse_and_validate(raw, expected_label=label)
            except Exception as exc:
                console.print(f"[yellow]⚠ Attempt {attempt}/3 failed for '{category}': {exc}[/yellow]")
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        return []

    @staticmethod
    def _parse_and_validate(raw: str, expected_label: int) -> list[dict]:
        raw = re.sub(r"```(?:json)?\s*", "", raw).rstrip("`")
        for i, ch in enumerate(raw):
            if ch != "[":
                continue
            depth = 0
            for j, c in enumerate(raw[i:], i):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[i:j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list) and parsed:
                                clean = []
                                for item in parsed:
                                    if isinstance(item, dict):
                                        inp = str(item.get("input",  "")).strip()
                                        out = str(item.get("output", "")).strip()
                                        try:
                                            lbl = int(item.get("label"))
                                        except (TypeError, ValueError):
                                            continue
                                        if (
                                            inp and out
                                            and lbl == expected_label
                                            and len(inp) >= 15
                                            and len(out) >= 5
                                        ):
                                            clean.append({"input": inp, "output": out, "label": lbl})
                                return clean
                        except json.JSONDecodeError:
                            break
        return []


def _deduplicate(examples: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for ex in examples:
        key = ex["input"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _distribute(total_needed: int, topics: list[tuple]) -> list[tuple[str, int, str, int]]:
    """
    Distribute `total_needed` examples across topics as evenly as possible.
    FIX: plain integer division left a systematic shortfall because
         800 // 7 == 114 and 114 * 7 == 798 (2 examples short every run).
         We now hand out the remainder one-by-one to the first N topics so
         the total always equals `total_needed` exactly.
    """
    n          = len(topics)
    base       = total_needed // n
    remainder  = total_needed % n
    result     = []
    for idx, (cat, lbl, desc) in enumerate(topics):
        count = base + (1 if idx < remainder else 0)
        if count > 0:
            result.append((cat, lbl, desc, count))
    return result


def generate() -> None:
    console.print(Panel.fit(
        "[bold cyan]Training Data Generator — MULTI-TURN AWARE[/bold cyan]",
        border_style="cyan",
    ))
    client = GeneratorClient()

    all_examples: list[dict]    = []
    existing_inputs: set[str]   = set()

    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ex = json.loads(line)
                        all_examples.append(ex)
                        existing_inputs.add(ex["input"].lower().strip())
                    except json.JSONDecodeError:
                        pass

    existing_high = sum(1 for e in all_examples if e["label"] == 1)
    existing_low  = sum(1 for e in all_examples if e["label"] == 0)
    need_high     = max(0, TARGET_HIGH - existing_high)
    need_low      = max(0, TARGET_LOW  - existing_low)

    if need_high == 0 and need_low == 0:
        console.print("[green]✅ Already have enough data.[/green]")
        return

    @dataclass
    class Job:
        category:  str
        label:     int
        description: str
        remaining: int

    jobs: list[Job] = []

    if need_high > 0:
        high_topics = HIGH_IMPORTANCE_TOPICS + MULTI_TURN_HIGH
        for cat, lbl, desc, count in _distribute(need_high, high_topics):
            jobs.append(Job(cat, lbl, desc, count))

    if need_low > 0:
        low_topics = LOW_IMPORTANCE_TOPICS + MULTI_TURN_LOW
        for cat, lbl, desc, count in _distribute(need_low, low_topics):
            jobs.append(Job(cat, lbl, desc, count))

    new_examples: list[dict] = []
    new_high = new_low = 0

    total_batches = sum(
        max(1, (j.remaining + BATCH_SIZE - 1) // BATCH_SIZE) for j in jobs
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating …", total=total_batches)

        # FIX: removed the ThreadPoolExecutor(max_workers=1) wrapper.
        # A pool with a single worker adds thread-management overhead while
        # providing zero concurrency benefit. Batches are now called directly.
        for job in jobs:
            fetched = 0
            while fetched < job.remaining:
                batch_n = min(BATCH_SIZE, job.remaining - fetched)
                progress.update(
                    task,
                    description=(
                        f"[cyan]{job.category}[/cyan] "
                        f"({'high' if job.label == 1 else 'low'})"
                    ),
                )

                batch = client.generate_batch(
                    job.category, job.label, job.description, batch_n
                )

                new_in_batch: list[dict] = []
                for ex in batch:
                    key = ex["input"].lower().strip()
                    if key not in existing_inputs:
                        existing_inputs.add(key)
                        new_examples.append(ex)
                        new_in_batch.append(ex)
                        if ex["label"] == 1:
                            new_high += 1
                        else:
                            new_low += 1

                if new_in_batch:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        for ex in new_in_batch:
                            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

                fetched += batch_n
                progress.advance(task)

    console.print(
        f"[green]✅ Added {len(new_examples)} new examples "
        f"({new_high} high, {new_low} low)[/green]"
    )
    all_examples = _deduplicate(all_examples + new_examples)
    console.print(Panel.fit(
        f"[bold]Total examples: {len(all_examples)}[/bold]\n"
        f"High: {sum(1 for e in all_examples if e['label'] == 1)} | "
        f"Low: {sum(1 for e in all_examples if e['label'] == 0)}",
        border_style="green",
    ))


if __name__ == "__main__":
    generate()
