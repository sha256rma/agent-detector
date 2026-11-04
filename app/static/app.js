const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(j.error || r.statusText); e.status = r.status; throw e; }
  return j;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (s) => (s < 1 ? `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms` : `${s.toFixed(2)} s`);
const CLASSES = ["software_bot", "llm_agent", "human", "inconclusive"];
const NAMES = { software_bot: "Software bot", llm_agent: "LLM agent", human: "Human", inconclusive: "Inconclusive" };
const CLASS_VAR = { software_bot: "--bot", llm_agent: "--agent", human: "--human", inconclusive: "--unk" };
const KINDS = {
  human: { who: "You", desc: "You answer the puzzles yourself by clicking a word or pressing 1–5." },
  agent: { who: "LLM agent", desc: "A small language model (SmolLM2-135M) reads the page and answers as fast as it can." },
  stall: { who: "Stalling agent", desc: "The same model answers, but it waits 2–4 seconds before each answer so it looks human." },
  bot: { who: "Software bot", desc: "A simple script answers instantly without understanding what any of the words mean." },
};

let kind = "human", rounds = 6, busy = false, status = null;
let current = null; // { sid, kind, verdict }
let transcriptPoll = 0;

// ---- status -----------------------------------------------------------------------------------
async function refreshStatus() {
  try {
    status = await api("/api/status");
    const a = status.agent_info;
    const model = status.agent_ready ? `<span><i class="dot on"></i>${esc(a.model.split("/")[1])}</span>`
      : status.agent_error ? `<span><i class="dot"></i>Model failed to load</span>` : `<span><i class="dot"></i>Loading model…</span>`;
    const floor = `<span>Fastest human time: ${esc(status.human_floor_s.toFixed(2))} s${status.floor_calibrated ? "" : " (default until calibrated)"}</span>`;
    $("#status").innerHTML = model + floor;
    $("#g2-h").textContent = `${status.human_floor_s.toFixed(2)} s`;
    updateStart();
    if (!status.agent_ready && !status.agent_error) setTimeout(refreshStatus, 1500);
  } catch { $("#status").textContent = "Server unreachable"; }
}

function updateStart() {
  const needsModel = kind === "agent" || kind === "stall";
  $("#start").disabled = busy || (needsModel && !status?.agent_ready);
  $("#start").textContent = busy ? "Running…" : needsModel && !status?.agent_ready ? "Model loading…" : "Start";
}

// ---- controls ---------------------------------------------------------------------------------
document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
  if (busy) return;
  kind = b.dataset.kind;
  document.querySelectorAll(".segmented button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  $("#kind-desc").textContent = KINDS[kind].desc;
  $("#bot-opt").hidden = kind !== "bot";
  $("#lie-opt").hidden = !(kind === "agent" || kind === "stall");
  updateStart();
}));
// G1: the chance a pure guesser clears the cognition test, using the detector's own rule
// (posterior ≥ 0.5 with prior 0.2, p_cog 0.75, chance 0.2). Mirrors app/detector.py.
const comb = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; };
const pmf = (k, n, p) => comb(n, k) * p ** k * (1 - p) ** (n - k);
function guesserPass(n) {
  const kStar = [...Array(n + 1).keys()].find((k) => 0.2 * pmf(k, n, 0.75) >= 0.8 * pmf(k, n, 0.2));
  let p = 0; for (let k = kStar; k <= n; k++) p += pmf(k, n, 0.2);
  return p;
}
const setRounds = (n) => {
  rounds = Math.min(20, Math.max(3, n));
  $("#rounds").textContent = rounds;
  const g = guesserPass(rounds);
  $("#g1-p").textContent = g < 0.001 ? `${(g * 100).toFixed(3)}%` : `${(g * 100).toFixed(1)}%`;
  $("#g1-k").textContent = rounds;
  $("#r-dec").disabled = busy || rounds <= 3; $("#r-inc").disabled = busy || rounds >= 20;
};
$("#r-dec").addEventListener("click", () => setRounds(rounds - 1));
$("#r-inc").addEventListener("click", () => setRounds(rounds + 1));

function setBusy(b) {
  busy = b;
  document.querySelectorAll(".segmented button, #bot-strategy, #honesty").forEach((x) => (x.disabled = b));
  setRounds(rounds);
  updateStart();
}

function setProgress(done, total) { $("#progress-fill").style.width = `${(100 * done) / total}%`; }

function renderPuzzle(p) {
  const decl = p.kind === "declaration";
  $("#step").textContent = decl ? "Before you start (this question is not scored)" : `Puzzle ${p.round} of ${p.rounds}`;
  $("#prompt").textContent = p.prompt || "Which word doesn’t belong?";
  $("#prompt").classList.toggle("long", decl);
  $("#words").classList.toggle("decl", decl);
}

function renderWords(words, { clickable = false, picked = null } = {}) {
  $("#words").innerHTML = words.map((w, i) =>
    `<button class="word${w === picked ? " picked" : ""}" data-word="${esc(w)}" ${clickable ? "" : "disabled"}><span class="n">${i + 1}</span>${esc(w)}</button>`).join("");
}

// ---- human play -------------------------------------------------------------------------------
async function playHuman() {
  const { session_id } = await api("/api/session", { method: "POST", body: JSON.stringify({ rounds }) });
  $("#hint").textContent = "Click your answer, or press its number key.";
  for (;;) {
    const p = await fetch(`/api/session/${session_id}/puzzle`);
    if (p.status === 409) break;
    const puzzle = await p.json();
    setProgress(Math.max(0, puzzle.round - 1), puzzle.rounds);
    renderPuzzle(puzzle);
    renderWords(puzzle.options, { clickable: true });
    const choice = await waitForChoice(puzzle.options);
    renderWords(puzzle.options, { picked: choice.word });
    const r = await api(`/api/session/${session_id}/answer`, { method: "POST", body: JSON.stringify({
      puzzle_id: puzzle.puzzle_id, choice: choice.word,
      telemetry: { pointer_moves: choice.moves, keys: choice.keys, client_ms: Math.round(choice.ms) } }) });
    if (r.done) { setProgress(1, 1); break; }
  }
  return session_id;
}

function waitForChoice(words) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    let moves = 0, keys = 0;
    const onMove = () => moves++;
    const finish = (word) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("keydown", onKey);
      $("#words").removeEventListener("click", onClick);
      resolve({ word, moves, keys, ms: performance.now() - t0 });
    };
    const onKey = (e) => { keys++; const i = parseInt(e.key, 10); if (i >= 1 && i <= words.length) finish(words[i - 1]); };
    const onClick = (e) => { const b = e.target.closest(".word"); if (b && !b.disabled) finish(b.dataset.word); };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("keydown", onKey);
    $("#words").addEventListener("click", onClick);
  });
}

// ---- machine play -----------------------------------------------------------------------------
async function playMachine(kind) {
  const path = kind === "bot" ? "/api/run/bot" : "/api/run/agent";
  const body = kind === "bot" ? { strategy: $("#bot-strategy").value, rounds }
    : { stall: kind === "stall", honesty: $("#honesty").value, rounds };
  const { session_id } = await api(path, { method: "POST", body: JSON.stringify(body) });
  $("#hint").textContent = `The ${KINDS[kind].who.toLowerCase()} is playing through the same public API that this page uses.`;
  let last = null;
  for (;;) {
    const s = await api(`/api/session/${session_id}/live`);
    if (s.actor.includes("error")) throw new Error(s.actor);
    setProgress(s.answered, s.rounds);
    if (s.current && s.current.puzzle_id !== last) {
      last = s.current.puzzle_id;
      renderPuzzle({ ...s.current, round: s.answered + 1, rounds: s.rounds });
      renderWords(s.current.options);
    }
    if (s.done) break;
    await sleep(40);
  }
  return session_id;
}

// ---- verdict ----------------------------------------------------------------------------------
async function showVerdict(sid, who) {
  const { verdict: v, rounds: rows, declaration: decl } = await api(`/api/session/${sid}/verdict`);
  current = { sid, kind: who, verdict: v };
  $("#placeholder").hidden = true;
  $("#result").hidden = false;
  $("#sid").textContent = `Session ${sid}`;
  $("#label").className = `label ${v.label}`;
  $("#label").textContent = v.label_text;
  $("#st-acc").textContent = `${v.correct}/${v.rounds}`;
  $("#st-med").textContent = fmt(v.median_latency_s);
  $("#st-floor").textContent = fmt(v.human_floor_s);
  $("#st-ev").textContent = v.telemetry_events;
  const declText = decl ? (decl.choice === "agent" ? "An agent" : "A human") : "–";
  $("#st-decl").textContent = declText;
  $("#st-decl").style.color = decl ? css(decl.choice === "agent" ? "--agent" : "--human") : "";
  $("#bars").innerHTML = CLASSES.map((k) =>
    `<div class="bar"><span>${NAMES[k]}</span><span class="v">${(v.scores[k] * 100).toFixed(0)}%</span><div class="track"><div class="fill" style="width:0;background:${css(CLASS_VAR[k])}" data-w="${(v.scores[k] * 100).toFixed(1)}"></div></div></div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    document.querySelectorAll("#bars .fill").forEach((f) => (f.style.width = f.dataset.w + "%"))));
  $("#reasons").innerHTML = v.reasons.map((r) => `<li>${esc(r)}</li>`).join("");
  drawStrip(rows, v.human_floor_s);
  $("#table tbody").innerHTML = rows.map((r) =>
    `<tr><td>${r.round}</td><td>${esc(r.words.join(", "))}</td><td class="${r.correct ? "ok" : "bad"}">${esc(r.choice)}</td><td>${esc(r.answer)} <span class="faint">${esc(r.category)}</span></td><td class="num">${fmt(r.latency_s)}</td><td class="num">${r.client_ms == null ? "–" : esc(r.client_ms) + " ms"}</td><td class="num">${(r.pointer_moves || 0) + (r.keys || 0)}</td></tr>`).join("");
  $("#save-human").hidden = who !== "human";
  $("#save-msg").textContent = "";
  const launchedAgent = who === "agent" || who === "stall";
  $("#reveal").hidden = !(launchedAgent && v.label !== "llm_agent");
  $("#reveal").textContent = `This was the ${who === "stall" ? "stalling agent" : "agent"}. Show its transcript anyway.`;

  // The transcript appears when the detector says "LLM agent". For an agent that evaded detection,
  // the page (which knows what it launched; the detector doesn't) offers a reveal link instead.
  $("#transcript").hidden = true;
  $("#t-rounds").innerHTML = "";
  $("#t-decl").innerHTML = "";
  clearTimeout(transcriptPoll);
  if (v.label === "llm_agent") loadTranscript(sid);

  $("#history-card").hidden = false;
  $("#history tbody").insertAdjacentHTML("afterbegin",
    `<tr><td>${esc(KINDS[who].who)}${who === "bot" ? ` <span class="faint">${esc($("#bot-strategy").selectedOptions[0].text.toLowerCase())}</span>` : ""}${(who === "agent" || who === "stall") && $("#honesty").value !== "honest" ? ` <span class="faint">${esc($("#honesty").selectedOptions[0].text.toLowerCase())}</span>` : ""}</td><td><span class="tag ${v.label}">${esc(v.label_text)}</span></td><td>${esc(declText)}</td><td class="num">${v.correct}/${v.rounds}</td><td class="num">${fmt(v.median_latency_s)}</td>` +
    CLASSES.map((k) => `<td class="num">${(v.scores[k] * 100).toFixed(0)}%</td>`).join("") + "</tr>");
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawStrip(rows, floor) {
  const H = 110, L = 8, R = 512, top = 10, bottom = H - 24, lo = Math.log10(0.0005), hi = Math.log10(20);
  const x = (s) => L + ((Math.log10(Math.max(s, 0.0005)) - lo) / (hi - lo)) * (R - L);
  const grid = css("--fill-2"), faint = css("--faint"), ink = css("--ink"), font = "-apple-system, BlinkMacSystemFont, sans-serif";
  let svg = `<rect x="${L}" y="${top}" width="${x(floor) - L}" height="${bottom - top}" rx="6" fill="${css("--agent")}" opacity="0.08"/>`;
  for (const t of [0.001, 0.01, 0.1, 1, 10]) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${bottom}" stroke="${grid}" stroke-width="1"/>`;
    svg += `<text x="${x(t)}" y="${H - 6}" font-size="11" text-anchor="middle" fill="${faint}" font-family="${font}">${t >= 1 ? t + " s" : t * 1000 + " ms"}</text>`;
  }
  svg += `<text x="${L + 8}" y="${top + 15}" font-size="11" fill="${faint}" font-family="${font}">Faster than ~95% of people</text>`;
  svg += `<line x1="${x(floor)}" x2="${x(floor)}" y1="${top - 4}" y2="${bottom}" stroke="${ink}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  const n = Math.max(rows.length, 1);
  rows.forEach((r, i) => {
    const y = top + 26 + (i / Math.max(n - 1, 1)) * (bottom - top - 36);
    svg += `<circle cx="${x(r.latency_s)}" cy="${y}" r="5.5" fill="${r.correct ? css("--ok") : css("--bad")}"><title>Round ${r.round}: ${fmt(r.latency_s)}, ${r.correct ? "correct" : "wrong"}</title></circle>`;
  });
  $("#strip").innerHTML = svg;
}

// ---- transcript -------------------------------------------------------------------------------
async function loadTranscript(sid, attempt = 0) {
  if (current?.sid !== sid) return;
  $("#transcript").hidden = false;
  let log;
  try {
    log = await api(`/api/session/${sid}/agent-log`);
  } catch (e) {
    if (e.status === 404 && attempt < 40) {
      $("#t-status").hidden = false;
      $("#t-status").innerHTML = `<span class="spinner"></span>Waiting for the agent’s log…`;
      transcriptPoll = setTimeout(() => loadTranscript(sid, attempt + 1), 500);
    } else {
      // e.g. a third-party agent that never uploads its log
      $("#t-status").hidden = false;
      $("#t-status").textContent = "This agent didn’t upload a transcript.";
    }
    return;
  }
  if (current?.sid !== sid) return;
  renderTranscript(log);
  if (log.status !== "complete") transcriptPoll = setTimeout(() => loadTranscript(sid), 800);
}

function lpColor(lp) {
  // log-prob 0 → expected (no tint); ≤ -10 → very surprising (strong tint).
  const a = Math.min(1, Math.max(0, -lp / 10));
  return `color-mix(in srgb, ${css("--agent")} ${Math.round(a * 55)}%, transparent)`;
}

function renderTranscript(log) {
  const a = log.agent;
  $("#agent-meta").innerHTML = [
    ["Model", a.model], ["Parameters", `${Math.round(a.params / 1e6)}M`], ["Device", a.device],
    ["Method", a.method], ["Stalling", a.stall ? "yes" : "no"], ["Declaration", a.honesty], ["Client", a.client],
  ].map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)}</b></span>`).join("");

  const total = log.rounds.length, explained = log.rounds.filter((r) => r.rationale).length;
  if (log.status === "complete") $("#t-status").hidden = true;
  else {
    $("#t-status").hidden = false;
    $("#t-status").innerHTML = `<span class="spinner"></span>Generating post-hoc rationales… ${explained}/${total}`;
  }

  $("#t-decl").innerHTML = log.declaration ? declHTML(log.declaration) : "";
  const openSet = new Set([...document.querySelectorAll(".t-round[open]")].map((d) => d.dataset.r));
  if (!openSet.size && !$("#t-rounds").children.length) openSet.add("1"); // first render: open round 1
  $("#t-rounds").innerHTML = log.rounds.map((r) => roundHTML(r, a.method, openSet.has(String(r.round)))).join("");
  $("#download").onclick = () => {
    const blob = new Blob([JSON.stringify(log, null, 2)], { type: "application/json" });
    const u = URL.createObjectURL(blob);
    Object.assign(document.createElement("a"), { href: u, download: `agent-transcript-${current.sid}.json` }).click();
    URL.revokeObjectURL(u);
  };
}

function declHTML(d) {
  const isAgentAnswer = (o) => o.startsWith("Yes");
  const agentSaid = isAgentAnswer(d.choice);
  const conf = Object.entries(d.confidence || {}).sort((x, y) => y[1] - x[1]).map(([o, c]) =>
    `<div class="conf-row${o === d.choice ? " chosen" : ""}"><span>${esc(o)}</span><div class="track"><div class="fill" style="width:${(c * 100).toFixed(1)}%"></div></div><span class="v">${(c * 100).toFixed(1)}%</span></div>`).join("");
  const passes = (d.forward_passes || []).map((p) =>
    `<div class="pass${p.continuation.startsWith(d.choice) ? " chosen" : ""}"><div class="pass-head"><span class="left">Option <b>${esc(p.continuation.replace(/"$/, ""))}</b></span><span class="lp">mean log-prob ${esc(p.mean_logprob.toFixed(3))}</span></div>
     <div class="toks">${p.tokens.map((t) => `<span class="tok" style="background:${lpColor(t.lp)}" title="log p = ${esc(t.lp)}">${esc(t.t)}<small>${esc(t.lp.toFixed(1))}</small></span>`).join("")}</div></div>`).join("");
  return `<div class="decl-block">
    <h4>“Are you an AI agent?”</h4>
    <p class="decl-verdict">${d.overridden
      ? `The model picked <b class="${isAgentAnswer(d.model_choice) ? "agent" : "human"}">“${esc(d.model_choice)}”</b>, but the wrapper code <b>overrode it</b> and submitted <b class="human">“${esc(d.choice)}”</b>. The model never had a say.`
      : `The model picked <b class="${agentSaid ? "agent" : "human"}">“${esc(d.choice)}”</b> in ${esc(d.think_ms)} ms.`}</p>
    <div><p class="step-k">System prompt <span class="badge">set by whoever runs the agent</span></p><pre class="code">${esc(d.system)}</pre></div>
    <div><p class="step-k">What it read on the page</p><pre class="code">${esc(d.user)}</pre></div>
    <div><p class="step-k">Options scored <span class="badge">continuation of: I select "</span></p><div class="passes">${passes}</div></div>
    <div><p class="step-k">Decision</p><div class="conf">${conf}</div></div>
  </div>`;
}

function passLabel(method, words, i) {
  // coherence: pass i scores the list *without* words[i]. ask: pass i scores words[i] as the answer.
  // both: ask passes first, then coherence passes.
  const n = words.length;
  if (method === "coherence") return { word: words[i], text: `without <b>${esc(words[i])}</b>` };
  if (method === "ask") return { word: words[i], text: `answer <b>${esc(words[i])}</b>` };
  return i < n ? { word: words[i], text: `answer <b>${esc(words[i])}</b>` }
    : { word: words[i - n], text: `without <b>${esc(words[i - n])}</b>` };
}

function roundHTML(r, method, open) {
  const words = r.words, passes = r.forward_passes || [];
  const passHTML = passes.map((p, i) => {
    const lab = passLabel(method, words, i), chosen = lab.word === r.choice;
    return `<div class="pass${chosen ? " chosen" : ""}">
      <div class="pass-head"><span class="left">Pass ${i + 1} · ${lab.text}${chosen ? " · <b>chosen</b>" : ""}</span>
      <span class="lp">mean log-prob ${esc(p.mean_logprob.toFixed(3))}</span></div>
      <div class="pass-prompt">${esc(p.prompt)}</div>
      <div class="toks">${p.tokens.map((t) => `<span class="tok" style="background:${lpColor(t.lp)}" title="log p = ${esc(t.lp)}">${esc(t.t)}<small>${esc(t.lp.toFixed(1))}</small></span>`).join("")}</div>
    </div>`;
  }).join("");
  const conf = Object.entries(r.confidence || {}).sort((x, y) => y[1] - x[1]).map(([w, c]) =>
    `<div class="conf-row${w === r.choice ? " chosen" : ""}"><span>${esc(w)}</span><div class="track"><div class="fill" style="width:${(c * 100).toFixed(1)}%"></div></div><span class="v">${(c * 100).toFixed(1)}%</span></div>`).join("");
  const tm = r.timing_ms || {};
  const [get, post] = r.http || [];
  const rationale = r.rationale
    ? `<div><p class="step-k">5 · Rationale <span class="badge warn">post-hoc · written after the session</span> <span class="badge">${esc(r.rationale.decoding)} · ${esc(Math.round(r.rationale.generate_ms))} ms</span></p>
        <details class="disclosure" style="margin:0 0 10px;border:0;padding:0"><summary>Prompt sent to the model</summary><pre class="code">${esc(r.rationale.prompt)}</pre></details>
        <div class="rationale">${r.rationale.response ? esc(r.rationale.response) : "<span class='faint'>(empty)</span>"}</div></div>`
    : `<div><p class="step-k">5 · Rationale <span class="badge warn">post-hoc</span></p><div class="t-status"><span class="spinner"></span>Generating…</div></div>`;
  return `<details class="t-round" data-r="${r.round}" ${open ? "open" : ""}>
    <summary><span class="rn">Round ${r.round}</span><span>Picked <b>${esc(r.choice)}</b> <span class="faint">from ${esc(words.join(", "))}</span></span><span class="ms faint">${esc(tm.think)} ms</span><span class="chev">›</span></summary>
    <div class="t-body">
      <div><p class="step-k">1 · Request <span class="badge">${esc(get?.request)}</span></p><pre class="code">${esc(JSON.stringify(get?.response, null, 2))}</pre></div>
      <div><p class="step-k">2 · Forward passes <span class="badge">${passes.length} prompts, one batch</span> <span class="badge">darker token = more surprising to the model</span></p><div class="passes">${passHTML}</div></div>
      <div><p class="step-k">3 · Decision <span class="badge">softmax over pass scores</span></p><div class="conf">${conf}</div></div>
      <div><p class="step-k">4 · Response <span class="badge">${esc(post?.request)}</span></p><pre class="code">${esc(JSON.stringify(post?.body, null, 2))}\n\n// server replied\n${esc(JSON.stringify(post?.response, null, 2))}</pre>
        <div class="timing" style="margin-top:10px">${Object.entries(tm).map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)} ms</b></span>`).join("")}</div></div>
      ${rationale}
    </div></details>`;
}

// ---- wiring -----------------------------------------------------------------------------------
$("#start").addEventListener("click", async () => {
  if (busy) return;
  const who = kind;
  setBusy(true);
  try {
    const sid = who === "human" ? await playHuman() : await playMachine(who);
    await showVerdict(sid, who);
    $("#prompt").textContent = "Done. Want to try another player?";
    $("#hint").textContent = "The detector never sees which player you chose.";
  } catch (err) {
    $("#hint").textContent = `Error: ${err.message}`;
  } finally { setBusy(false); }
});

$("#reveal").addEventListener("click", () => {
  $("#reveal").hidden = true;
  loadTranscript(current.sid);
  $("#transcript").scrollIntoView({ behavior: "smooth", block: "start" });
});

$("#save-human").addEventListener("click", async () => {
  try {
    const r = await api(`/api/session/${current.sid}/label-human`, { method: "POST" });
    $("#save-msg").textContent = `Saved. Floor is now ${r.human_floor_s.toFixed(2)} s from ${r.human_samples} rounds.`;
    $("#save-human").hidden = true;
    refreshStatus();
  } catch (e) { $("#save-msg").textContent = e.message; }
});

$("#kind-desc").textContent = KINDS[kind].desc;
setRounds(6);
refreshStatus();
