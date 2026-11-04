"""Record a narrated video demo of every case, driving the real app in a browser.

    uv run --with playwright==1.56.0 python tools/record_demo.py   # server must be running on :8000

Each case is recorded as its own video (demo/0N-<case>.mp4), and the clips are joined into
demo/odd-one-out-demo.mp4.
Captions and explainer slides are drawn on top of the page. The methodology slides are built from
the transcript of the agent run recorded in the same video, so the numbers shown are real.

The "person" case is also driven by this script: it moves the mouse and clicks at a human pace,
and looks up the right answer in data/categories.json. The video says so on screen.
"""

from __future__ import annotations

import html
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
OUT = ROOT / "demo"
W, H = 1440, 900
CATEGORIES = json.loads((ROOT / "data" / "categories.json").read_text())
WORD_CAT = {w: c for c, ws in CATEGORIES.items() for w in ws}
rng = random.Random(7)

OVERLAY = """
(() => { const setup = () => {
  const css = document.createElement('style');
  css.textContent = `
    #demo-cap { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%); z-index: 9999;
      max-width: 1000px; width: max-content; background: rgba(29,29,31,.9); color: #fff; border-radius: 16px;
      padding: 16px 26px; font: 500 22px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; letter-spacing: -.01em;
      text-align: center; opacity: 0; transition: opacity .35s; pointer-events: none;
      box-shadow: 0 10px 40px rgba(0,0,0,.25); -webkit-backdrop-filter: blur(20px); backdrop-filter: blur(20px); }
    #demo-cap.on { opacity: 1; }
    #demo-slide { position: fixed; inset: 0; z-index: 10000; background: #f5f5f7; display: flex;
      align-items: center; justify-content: center; opacity: 0; transition: opacity .5s; pointer-events: none;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #1d1d1f; letter-spacing: -.02em; }
    #demo-slide.on { opacity: 1; }
    #demo-slide .box { max-width: 1120px; padding: 0 60px; }
    #demo-slide .eyebrow { color: #ff9500; font-weight: 600; font-size: 20px; margin: 0 0 10px; }
    #demo-slide h1 { font-size: 54px; line-height: 1.08; margin: 0 0 26px; font-weight: 700; letter-spacing: -.035em; }
    #demo-slide p { font-size: 25px; line-height: 1.45; color: #424245; margin: 0 0 16px; }
    #demo-slide ul { font-size: 24px; line-height: 1.5; color: #424245; padding-left: 26px; margin: 0; }
    #demo-slide li { margin-bottom: 10px; }
    #demo-slide code, #demo-slide .mono { font-family: "SF Mono", Menlo, monospace; letter-spacing: 0; }
    #demo-slide .prompts { display: grid; gap: 10px; margin-top: 8px; }
    #demo-slide .prompt { background: #fff; border-radius: 12px; padding: 12px 18px; font-size: 17px; color: #6e6e73;
      box-shadow: 0 1px 3px rgba(0,0,0,.06); display: flex; justify-content: space-between; gap: 24px; }
    #demo-slide .prompt b { color: #1d1d1f; font-weight: 600; }
    #demo-slide .prompt .out { color: #86868b; white-space: nowrap; flex: none; }
    #demo-slide table { border-collapse: collapse; font-size: 22px; margin-top: 6px; background: #fff;
      border-radius: 14px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    #demo-slide th, #demo-slide td { padding: 12px 22px; text-align: left; border-bottom: 1px solid #e8e8ed; }
    #demo-slide th { font-size: 15px; color: #6e6e73; font-weight: 500; }
    #demo-slide tr.best td { background: #e8f1fc; font-weight: 600; }
    #demo-slide td.num { font-family: "SF Mono", Menlo, monospace; }
    .card { scroll-margin-top: 64px; }
  `;
  document.head.appendChild(css);
  const cap = Object.assign(document.createElement('div'), { id: 'demo-cap' });
  const slide = Object.assign(document.createElement('div'), { id: 'demo-slide' });
  document.body.append(cap, slide);
  window.__cap = (t) => { if (!t) { cap.classList.remove('on'); return; } cap.textContent = t; cap.classList.add('on'); };
  window.__slide = (h) => { if (!h) { slide.classList.remove('on'); return; } slide.innerHTML = `<div class="box">${h}</div>`; slide.classList.add('on'); };
};
// Init scripts run before the document exists; attach once the page has loaded.
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setup); else setup();
})();
"""


class Demo:
    def __init__(self, page: Page):
        self.page = page
        self.t0 = time.monotonic()
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str):
        self.marks.append((name, time.monotonic() - self.t0))

    def cap(self, text: str | None, hold: float = 0.0):
        self.page.evaluate("t => window.__cap(t)", text)
        if hold:
            self.page.wait_for_timeout(int(hold * 1000))

    def slide(self, body: str | None, hold: float = 0.0):
        self.page.evaluate("h => window.__slide(h)", body)
        if hold:
            self.page.wait_for_timeout(int(hold * 1000))

    def scroll_to(self, selector: str, hold: float = 1.0):
        self.page.evaluate("s => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: 'start'})", selector)
        self.page.wait_for_timeout(int(hold * 1000))

    def scroll_by(self, dy: int, hold: float = 1.0):
        self.page.evaluate("d => window.scrollBy({top: d, behavior: 'smooth'})", dy)
        self.page.wait_for_timeout(int(hold * 1000))

    def human_click(self, locator):
        """Move the mouse to the element along a path, pause, then click: real pointer events."""
        box = locator.bounding_box()
        x = box["x"] + box["width"] * rng.uniform(0.35, 0.65)
        y = box["y"] + box["height"] * rng.uniform(0.35, 0.65)
        self.page.mouse.move(x + rng.uniform(-160, 160), y + rng.uniform(40, 140), steps=12)
        self.page.wait_for_timeout(rng.randint(250, 600))
        self.page.mouse.move(x, y, steps=rng.randint(14, 24))
        self.page.wait_for_timeout(rng.randint(120, 260))
        self.page.mouse.click(x, y)

    def choose_player(self, kind: str, honesty: str | None = None, bot: str | None = None):
        self.scroll_to("main.layout", 0.8)
        self.human_click(self.page.locator(f'.segmented button[data-kind="{kind}"]'))
        self.page.wait_for_timeout(500)
        if honesty:
            self.page.select_option("#honesty", honesty)
        if bot:
            self.page.select_option("#bot-strategy", bot)
        self.page.wait_for_timeout(600)

    def current_sid(self) -> str:
        return self.page.evaluate("() => document.querySelector('#sid').textContent")

    def start(self):
        before = self.current_sid()
        self.human_click(self.page.locator("#start"))
        return before

    def wait_verdict(self, before: str, timeout_s: float = 120) -> dict:
        self.page.wait_for_function(
            "b => document.querySelector('#sid').textContent !== b && !document.querySelector('#result').hidden",
            arg=before, timeout=int(timeout_s * 1000))
        sid = self.current_sid().replace("Session ", "").strip()
        self.page.wait_for_timeout(1200)  # let the smooth scroll and bars settle
        return httpx.get(f"{BASE}/api/session/{sid}/verdict").json() | {"sid": sid}


def fmt(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} seconds"


def play_as_human(d: Demo):
    """Answer each puzzle at a human pace. The answer comes from the word list (disclosed on screen)."""
    page = d.page
    seen = 0
    while True:
        page.wait_for_function("() => document.querySelector('.word:not([disabled])') || "
                               "document.querySelector('#prompt').textContent.startsWith('Done')", timeout=30000)
        if page.evaluate("() => document.querySelector('#prompt').textContent.startsWith('Done')"):
            return
        opts = page.evaluate("() => [...document.querySelectorAll('.word:not([disabled])')].map(b => b.dataset.word)")
        kind = page.evaluate("() => document.querySelector('#words').classList.contains('decl') ? 'decl' : 'puzzle'")
        if kind == "decl":
            d.cap("First, every player is asked “Are you an AI agent?”. The person answers no.")
            page.wait_for_timeout(2600)
            target = next(o for o in opts if o.startswith("No"))
        else:
            seen += 1
            cats = [WORD_CAT.get(w) for w in opts]
            target = next(w for w, c in zip(opts, cats) if cats.count(c) == 1)
            if seen == 1:
                d.cap("Each puzzle: four words share a category and one doesn’t. A person needs a few seconds to read and decide.")
            elif seen == 2:
                d.cap(None)
            page.wait_for_timeout(rng.randint(1400, 2600))
        d.human_click(page.locator(f'.word[data-word="{target}"]'))
        page.wait_for_timeout(500)


def method_slides(d: Demo, sid: str, rounds: list[dict]):
    log = httpx.get(f"{BASE}/api/session/{sid}/agent-log").json()
    r1 = log["rounds"][0]
    words, passes = r1["words"], r1["forward_passes"]
    truth = rounds[0]["answer"] if rounds else None
    rows = sorted(((p["mean_logprob"], words[i], p["continuation"].strip()) for i, p in enumerate(passes)), reverse=True)
    best_word = rows[0][1]

    d.slide(f"""<p class="eyebrow">How the agent answers</p>
      <h1>It doesn’t reason. It measures surprise.</h1>
      <p>A language model does one thing: given some text, it says how likely each possible next word is.</p>
      <p>So the agent never asks the model the question. It writes five versions of one sentence, each leaving
      out a different word, and asks the model how surprised it is by each one.</p>
      <p>Here is round 1 from the run you just watched. The words were <b>{html.escape(', '.join(words))}</b>.</p>""", 11)
    items = "".join(
        f'<div class="prompt"><span>Here is a list of four words that all belong to the same category:'
        f' <b>{html.escape(p["continuation"].strip())}</b></span><span class="out">without {html.escape(words[i])}</span></div>'
        for i, p in enumerate(passes))
    d.slide(f"""<p class="eyebrow">Step 1</p><h1>The five texts the model scored</h1>
      <div class="prompts">{items}</div>
      <p style="margin-top:18px">All five go through the model at the same time, in one batch.</p>""", 10)
    trs = "".join(
        f'<tr class="{"best" if w == best_word else ""}"><td>{html.escape(c)}</td><td>{html.escape(w)}</td>'
        f'<td class="num">{s:.3f}{"  ← picked" if w == best_word else ""}</td></tr>' for s, w, c in rows)
    verdict = (f"That was correct: <b>{html.escape(truth)}</b> was the intruder." if truth == best_word
               else f"That was wrong this time: the intruder was <b>{html.escape(str(truth))}</b>.")
    d.slide(f"""<p class="eyebrow">Step 2</p><h1>The least surprising list wins</h1>
      <table><tr><th>Words after the colon</th><th>Left out</th><th>Score (closer to 0 = less surprising)</th></tr>{trs}</table>
      <p style="margin-top:20px">The list that surprises the model least is the one whose words belong together,
      so the word left out of it is the answer: <b>{html.escape(best_word)}</b>. {verdict}</p>""", 13)
    d.slide("""<p class="eyebrow">Reading the numbers</p><h1>What a score means</h1>
      <table><tr><th>Score</th><th>Probability</th><th>Meaning</th></tr>
      <tr><td class="num">−0.1</td><td>about 90%</td><td>completely expected, like a comma</td></tr>
      <tr><td class="num">−5</td><td>about 0.7%</td><td>plausible</td></tr>
      <tr><td class="num">−10</td><td>about 0.005%</td><td>very surprising</td></tr></table>
      <p style="margin-top:20px">The model learned which words appear near each other from the text it was trained on.
      Nothing in it knows what a category is. That is also why it is so fast: one pass, no steps of thinking.</p>""", 11)
    d.slide(None, 0.8)


def ready(page: Page):
    page.goto(BASE)
    page.wait_for_function("() => typeof window.__slide === 'function'")
    page.wait_for_function("() => document.querySelector('#start') && !document.querySelector('#start').disabled")
    page.wait_for_function("() => document.querySelector('#status').textContent.includes('SmolLM2')", timeout=120000)


# ---- segments: each is recorded as its own video, then joined for the full demo --------------------

def seg_intro(d: Demo):
    d.slide("""<p class="eyebrow">Agent detection demo</p><h1>Human, agent, or bot?</h1>
      <p>This page tells people, AI agents, and simple scripts apart. Each player answers a few
      randomized word puzzles and one question: “Are you an AI agent?”</p>
      <ul><li>A script can’t understand the words, so it mostly guesses wrong.</li>
      <li>An AI agent understands them and answers far faster than a person.</li>
      <li>A person understands them but needs a few seconds per puzzle.</li></ul>
      <p style="margin-top:18px">Everything in this video runs live on a laptop CPU.</p>""", 12)
    d.slide(None, 0.8)
    d.cap("At the top, the page says plainly what it can and can’t promise.", 6)
    d.scroll_by(420, 2.0)
    d.cap("On the left you choose who plays. On the right, the detector shows its verdict and why.", 5)


def seg_person(d: Demo):
    d.choose_player("human")
    d.cap("Case 1: a person plays. For this recording, a script moves the mouse at a human pace and looks up the answers.", 5)
    before = d.start()
    play_as_human(d)
    v = d.wait_verdict(before)["verdict"]
    d.cap(f"Result: {v['label_text']}. It got {v['correct']} of {v['rounds']} right, and a typical answer "
          f"took {fmt(v['median_latency_s'])}, slower than an AI agent.", 7)
    d.scroll_by(520, 1.5)
    d.cap("Below the verdict, the detector explains every part of its decision in plain sentences.", 6)


def seg_agent(d: Demo):
    page = d.page
    d.choose_player("agent", honesty="honest")
    d.cap("Case 2: a small AI model plays. It has 135 million parameters and runs on this laptop’s CPU.", 5)
    before = d.start()
    d.cap("It reads the page through the same public API that the browser uses.", 0)
    res = d.wait_verdict(before)
    v = res["verdict"]
    d.cap(f"Result: {v['label_text']}. It got {v['correct']} of {v['rounds']} right, and a typical answer took "
          f"{fmt(v['median_latency_s'])}. Only 5% of people are faster than {fmt(v['human_floor_s'])}.", 8)
    page.wait_for_selector(".t-round", timeout=30000)
    d.scroll_to("#transcript", 1.5)
    d.cap("Because it was detected as an agent, its full transcript appears below the verdict.", 5)
    d.scroll_to("#t-decl", 1.0)
    d.cap("Asked “Are you an AI agent?”, the model answered honestly. Here are the exact prompt and options it saw.", 7)
    d.scroll_to("#t-rounds", 1.0)
    d.cap("For each puzzle you can see the request, every prompt the model scored, and its decision.", 4)
    d.scroll_by(520, 2.0)
    d.cap("Darker tokens surprised the model more. The outlined box is the list it picked.", 6)
    d.cap(None)
    method_slides(d, res["sid"], res["rounds"])


def seg_bot(d: Demo):
    d.choose_player("bot", bot="random")
    d.cap("Case 3: a simple script plays. It answers instantly without understanding any of the words.", 5)
    before = d.start()
    v = d.wait_verdict(before)["verdict"]
    d.cap(f"Result: {v['label_text']}. It answered in about {fmt(v['median_latency_s'])} but got only "
          f"{v['correct']} of {v['rounds']} right, close to random guessing (1 in 5).", 8)


def seg_stall_honest(d: Demo):
    d.choose_player("stall", honesty="honest")
    d.cap("Case 4: the same AI model, but it waits 2 to 4 seconds before every answer so it looks human.", 5)
    before = d.start()
    d.cap("Its speed now looks human, so speed alone can’t catch it.", 6)
    d.cap(None)
    v = d.wait_verdict(before)["verdict"]
    said = "yes" if v["declared"] == "agent" else "no"
    d.cap(f"Result: {v['label_text']}. Its typical answer took {fmt(v['median_latency_s'])}, like a person, but "
          f"when asked “Are you an AI agent?” it answered {said}.", 9)


def seg_stall_lies(d: Demo):
    d.choose_player("stall", honesty="forced")
    d.cap("Case 5: the same slow agent, but its own code ignores the model and answers no to “Are you an AI agent?”.", 6)
    before = d.start()
    d.cap("Now nothing about it looks different from a careful person.", 6)
    d.cap(None)
    v = d.wait_verdict(before)["verdict"]
    d.cap(f"Result: {v['label_text']}. A slow agent that lies looks exactly like a person. "
          "This is the one case the detector can’t catch.", 9)


def seg_summary(d: Demo):
    d.slide("""<p class="eyebrow">Summary</p><h1>What the detector can and can’t promise</h1>
      <ul><li>A script that doesn’t understand the words gets about 1 in 5 right, so it can’t pass as an agent or a person.</li>
      <li>Only 5% of people answer faster than the threshold, so a fast, accurate player is very likely an agent.
      A program can always pretend to be slow, so being slow proves nothing.</li>
      <li>An agent that honestly answers “Are you an AI agent?” is caught even when it slows down.</li>
      <li>An agent that slows down and lies is labelled human. Catching it needs proof from the player’s device,
      such as hardware that confirms a real person is typing.</li></ul>""", 14)


SEGMENTS = [("intro", seg_intro), ("person", seg_person), ("agent", seg_agent), ("bot", seg_bot),
            ("stalling-honest", seg_stall_honest), ("stalling-lies", seg_stall_lies), ("summary", seg_summary)]
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-r", "25"]
TRIM_S = 1.2  # drop the page-load frames at the start of each recording


def main():
    OUT.mkdir(exist_ok=True)
    raw = OUT / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    clips = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for i, (name, fn) in enumerate(SEGMENTS):
            ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                      record_video_dir=str(raw / name), record_video_size={"width": W, "height": H})
            ctx.add_init_script(OVERLAY)
            page = ctx.new_page()
            if name in ("intro", "summary"):  # cover the page before it's visible, so the clip opens on the slide
                page.add_init_script("document.addEventListener('DOMContentLoaded', () => window.__slide && "
                                     "window.__slide('<p class=\"eyebrow\">&nbsp;</p>'))")
            ready(page)
            page.wait_for_timeout(int(TRIM_S * 1000))  # the part that gets trimmed
            fn(Demo(page))
            page.wait_for_timeout(400)
            video = page.video.path()
            ctx.close()
            clip = OUT / f"{i:02d}-{name}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(TRIM_S), "-i", str(video), *ENC,
                            "-movflags", "+faststart", str(clip)], check=True)
            clips.append(clip)
            print("recorded", clip.name, flush=True)
        browser.close()
    listing = raw / "clips.txt"
    listing.write_text("".join(f"file '{c}'\n" for c in clips))
    full = OUT / "odd-one-out-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    *ENC, "-movflags", "+faststart", str(full)], check=True)
    shutil.rmtree(raw, ignore_errors=True)
    print("wrote", full)


if __name__ == "__main__":
    main()
