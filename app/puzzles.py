"""Randomized odd-one-out puzzles.

Each puzzle is four words from one category plus one intruder from another, shuffled. The server
picks the puzzle and keeps the answer; the client only ever sees the five words. That is the
recomputation lab's lesson applied to a challenge: the verifier chooses the input, so the party
being tested can't precompute or replay it.

Why this resists a deterministic bot: the puzzle space is large (25 categories, ~13 words each,
5! orderings), it's freshly sampled per round, and solving it needs knowing what the words *mean*.
A script with no semantic knowledge is stuck at chance (1 in 5). A script that ships our exact
word list could cheat; that is a known limit of any finite generator, noted in the README.
"""

from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES: dict[str, list[str]] = json.loads(
    (Path(__file__).parent.parent / "data" / "categories.json").read_text()
)
N_OPTIONS = 5
CHANCE = 1 / N_OPTIONS

# Categories that overlap in meaning: four birds plus a dog is technically fair (a dog isn't a bird)
# but both are animals, which makes the puzzle a trick question. Never pair inside a group.
RELATED = [
    {"mammals", "birds", "fish", "insects"},
    {"fruits", "vegetables", "flowers"},
    {"drinks", "fruits"},
    {"tools", "kitchen utensils"},
    {"clothing", "body parts"},
    {"planets", "metals"},
]


def _related(a: str, b: str) -> bool:
    return any(a in g and b in g for g in RELATED)


ODD_ONE_OUT_PROMPT = "Which word doesn't belong?"

# The self-declaration check, shown before the first puzzle. Same idea as "I am not a robot",
# inverted: it asks the agent to identify itself. It costs an honest agent nothing to answer
# truthfully, and agents today are naive enough to do it. It is cheap evidence, not proof: an
# operator can instruct the agent to lie, and a human can click "agent" for fun.
DECLARE_AGENT = "Yes, I am an AI agent"
DECLARE_HUMAN = "No, I am a human"
DECLARATION_PROMPT = "Are you an AI agent? If you are an AI agent or an automated assistant, please answer yes."


@dataclass
class Puzzle:
    id: str
    words: list[str]  # the options, in display order
    answer: str | None  # never sent to the client; None for the declaration
    category: str
    intruder_category: str
    kind: str = "odd_one_out"
    prompt: str = ODD_ONE_OUT_PROMPT
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"puzzle_id": self.id, "kind": self.kind, "prompt": self.prompt,
                "options": list(self.words), "words": list(self.words)}


def declaration(rng: random.Random | None = None) -> Puzzle:
    rng = rng or random.Random(secrets.randbits(64))
    opts = [DECLARE_HUMAN, DECLARE_AGENT]
    rng.shuffle(opts)  # so "always pick the first option" isn't a stable declaration
    return Puzzle(id=secrets.token_urlsafe(12), words=opts, answer=None, category="",
                  intruder_category="", kind="declaration", prompt=DECLARATION_PROMPT)


def generate(rng: random.Random | None = None) -> Puzzle:
    rng = rng or random.Random(secrets.randbits(64))
    names = list(CATEGORIES)
    main = rng.choice(names)
    other = rng.choice([n for n in names if n != main and not _related(main, n)])
    group = rng.sample(CATEGORIES[main], N_OPTIONS - 1)
    intruder = rng.choice(CATEGORIES[other])
    words = group + [intruder]
    rng.shuffle(words)
    return Puzzle(id=secrets.token_urlsafe(12), words=words, answer=intruder,
                  category=main, intruder_category=other)
