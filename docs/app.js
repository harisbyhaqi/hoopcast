const state = { data: null, month: null, day: null, team: "", updates: "all" };

const STATUS_LABEL = { out: "Out", doubtful: "Doubtful", questionable: "Questionable", "day-to-day": "Day-to-day" };
// Update filter chip → change kinds, pill label, pill class
const CHANGE_TYPES = {
  injury: { kinds: ["injury_new"], chip: "Injuries" },
  status: { kinds: ["status_change"], chip: "Status changes" },
  signed: { kinds: ["signed"], chip: "Signings" },
  released: { kinds: ["released"], chip: "Releases" },
  traded: { kinds: ["traded"], chip: "Trades" },
  cleared: { kinds: ["cleared"], chip: "Cleared" },
};
const CHANGE_PILL = {
  injury_new: ["New injury", "k-injury"], status_change: ["Status change", "k-status"],
  signed: ["Signed", "k-good"], cleared: ["Cleared", "k-good"],
  released: ["Released", "k-neutral"], traded: ["Traded", "k-trade"],
};
const STATUS_PILL = { out: "k-injury", doubtful: "k-injury", questionable: "k-status", "day-to-day": "k-neutral" };

async function loadSeason() {
  const res = await fetch("season.json?v=" + Date.now());
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const pad = n => String(n).padStart(2, "0");
const parseDay = iso => new Date(iso + "T12:00:00");   // avoid TZ shift on date-only strings
const monthKey = iso => iso.slice(0, 7);
const ET = { timeZone: "America/New_York" };
const etDay = iso => new Date(iso).toLocaleDateString("en-CA", ET);   // YYYY-MM-DD in ET
const fmtTip = iso => new Date(iso).toLocaleTimeString("en-US", { ...ET, hour: "numeric", minute: "2-digit" }) + " ET";
const fmtClock = iso => new Date(iso).toLocaleTimeString("en-US", { ...ET, hour: "numeric", minute: "2-digit" });
const fmtShort = iso => parseDay(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
const signed = (v, digits = 1) => {
  const r = +v.toFixed(digits);
  return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(digits);
};

function tile(abbr, size = "") {
  const t = state.data.teams[abbr] || {};
  return `<div class="tile ${size}">${esc(abbr)}<img src="${esc(t.logo)}" alt="${esc(t.name || abbr)}" loading="lazy" onerror="this.remove()"></div>`;
}

function visibleGames() {
  const t = state.team;
  return state.data.games.filter(g => !t || g.home === t || g.away === t);
}

function projRecord(abbr) {
  const s = state.data.standings;
  const r = [...s.East, ...s.West].find(x => x.team === abbr);
  return r ? `${Math.round(r.proj_w)}-${Math.round(r.proj_l)}` : "–";
}

// ── Header ──
function renderHeader(d) {
  document.getElementById("season-chip").textContent = "NBA " + d.season.replace("-", "–");
  document.getElementById("run-label").textContent = "Model run · " + new Date(d.generated).toLocaleString("en-US",
    { ...ET, month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET";
  const recent = d.changes.filter(c => Date.parse(d.generated) - Date.parse(c.time) < 864e5).length;
  const badge = document.getElementById("updates-badge");
  badge.textContent = recent;
  badge.hidden = !recent;
  document.getElementById("model-note").textContent =
    `Projections blend each roster's DARKO player ratings with this season's form and today's injury reports. ` +
    `Win probability comes from the projected margin plus a ${d.context.hca.toFixed(1)}-point home edge.`;
}

// ── Calendar ──
function renderCalendar() {
  const [y, m] = state.month.split("-").map(Number);
  const first = new Date(y, m - 1, 1);
  document.getElementById("cal-month").textContent = first.toLocaleDateString("en-US", { month: "long", year: "numeric" });

  const byDay = {};
  visibleGames().forEach(g => (byDay[g.date] ??= []).push(g));

  const cells = [];
  for (let i = 0; i < first.getDay(); i++) cells.push(`<div class="cal-cell empty"></div>`);
  const daysInMonth = new Date(y, m, 0).getDate();
  for (let day = 1; day <= daysInMonth; day++) {
    const iso = `${state.month}-${pad(day)}`;
    const n = (byDay[iso] || []).length;
    const classes = ["cal-cell"];
    if (n) classes.push("has-games");
    if (iso === state.day) classes.push("selected");
    if (iso === state.data.today) classes.push("today");
    const count = n ? (state.team ? "●" : n + "g") : "";
    cells.push(`<button class="${classes.join(" ")}" data-day="${iso}">
      <span class="cal-num">${day}</span><span class="cal-count">${count}</span></button>`);
  }
  const grid = document.getElementById("cal-grid");
  grid.innerHTML = cells.join("");
  grid.querySelectorAll("button.cal-cell").forEach(b => b.addEventListener("click", () => {
    state.day = b.dataset.day;
    renderCalendar();
    renderDay();
  }));
}

// ── Game cards ──
function confidence(pHome) {
  const p = Math.max(pHome, 100 - pHome);
  return p < 56 ? ["Toss-up", "conf-tossup"] : p < 66 ? ["Lean", "conf-lean"] : ["Strong", "conf-strong"];
}

function basisLabel(pred) {
  if (!pred || pred.form_weight == null) return "";
  const form = Math.round(pred.form_weight * 100);
  return form === 0 ? "Roster-based" : `${100 - form}% roster · ${form}% form`;
}

function lineupNotes(g) {
  const inj = g.injuries;
  if (!inj) return "";
  const notes = [[g.away, inj.away, inj.away_delta], [g.home, inj.home, inj.home_delta]]
    .filter(([, list]) => list && list.length)
    .map(([abbr, list, delta]) => `
      <div class="note">
        <div class="note-line"><span class="note-head">${esc(abbr)} lineup impact</span><span class="note-impact">${signed(delta || 0)} pts</span></div>
        ${list.map(p => `<div class="note-line"><span class="note-player">${esc(p.player)}</span>
          <span class="note-status ${p.status === "out" ? "out" : ""}">${esc(STATUS_LABEL[p.status] || p.status)}</span></div>`).join("")}
      </div>`).join("");
  return notes ? `<div class="notes">${notes}</div>` : "";
}

function renderCard(g) {
  const final = !!g.result;
  const probs = final ? g.frozen || g.pred : g.pred;
  const pHome = probs ? probs.p_home : null;
  const score = final ? g.result : g.pred;
  // Final games highlight the winner; upcoming ones highlight the favorite
  const homeFav = final ? g.result.home > g.result.away : pHome != null && pHome >= 50;

  let chip = basisLabel(g.pred);
  let chipClass = "chip";
  if (final && g.frozen) {
    const right = (g.frozen.p_home >= 50) === (g.result.home > g.result.away);
    // Same per-team error as the Accuracy tab's "avg pts off"
    const off = (Math.abs(g.frozen.home - g.result.home) + Math.abs(g.frozen.away - g.result.away)) / 2;
    chip = `${right ? "Called it" : "Missed"} · off ${off.toFixed(1)}`;
    chipClass = "pill " + (right ? "verdict-right" : "verdict-wrong");
  }

  const row = (abbr, side, pts, p, fav) => `
    <div class="team-row ${fav ? "fav" : ""}">
      ${tile(abbr)}
      <div style="min-width:0">
        <div class="team-name">${esc(state.data.teams[abbr]?.name || abbr)}</div>
        <div class="team-side">${side} · proj ${projRecord(abbr)}</div>
      </div>
      <div class="team-score">${pts == null ? "–" : final ? pts : pts.toFixed(1)}</div>
      <div class="team-pct">${p == null ? "" : p.toFixed(1) + "%"}</div>
    </div>`;

  let foot = "";
  if (pHome != null) {
    const [conf, confClass] = confidence(pHome);
    const margin = Math.abs(probs.home - probs.away).toFixed(1);
    foot = `
      <div class="prob"><div class="prob-bar">
        <div class="${pHome < 50 ? "fav" : ""}" style="width:${(100 - pHome).toFixed(1)}%"></div>
        <div class="${pHome >= 50 ? "fav" : ""}"></div>
      </div></div>
      <div class="game-foot">
        <div class="pick"><span class="pick-k">${final ? "Picked" : "Pick"}</span>
          <span class="pick-team">${esc(pHome >= 50 ? g.home : g.away)}</span><span class="pick-by">by ${margin}</span></div>
        <span class="pill ${confClass}">${conf}</span>
      </div>`;
  }

  return `
    <article class="card game">
      <div class="game-top">
        <span class="game-time">${final ? "Final" + (g.frozen ? ` · proj ${g.frozen.away.toFixed(0)}–${g.frozen.home.toFixed(0)}` : "") : esc(fmtTip(g.tip))}${g.type === "playoffs" ? " · Playoffs" : ""}</span>
        ${chip ? `<span class="${chipClass}">${esc(chip)}</span>` : ""}
      </div>
      <div class="game-rows">
        ${row(g.away, "Away", score?.away, pHome == null ? null : 100 - pHome, !homeFav)}
        ${row(g.home, "Home", score?.home, pHome, homeFav)}
      </div>
      ${foot}
      ${final ? "" : lineupNotes(g)}
    </article>`;
}

function renderDay() {
  const games = visibleGames().filter(g => g.date === state.day);
  const d = parseDay(state.day);
  document.getElementById("day-eyebrow").textContent =
    `${d.toLocaleDateString("en-US", { weekday: "long" })} · ${d.getFullYear()}`;
  document.getElementById("day-label").textContent = d.toLocaleDateString("en-US", { month: "long", day: "numeric" });
  document.getElementById("today-pill").hidden = state.day !== state.data.today;

  const priced = games.filter(g => g.pred || g.frozen);
  const p = g => (g.pred || g.frozen).p_home;
  let summary = "";
  if (games.length) {
    const stats = [["Games", games.length], ["First tip", fmtTip(games.map(g => g.tip).sort()[0])]];
    if (priced.length) {
      const close = priced.reduce((b, g) => Math.abs(p(g) - 50) < Math.abs(p(b) - 50) ? g : b);
      const big = priced.reduce((b, g) => Math.abs(p(g) - 50) > Math.abs(p(b) - 50) ? g : b);
      stats.push(["Closest", `${close.away} @ ${close.home}`],
                 ["Most confident", `${p(big) >= 50 ? big.home : big.away} ${Math.round(Math.max(p(big), 100 - p(big)))}%`]);
    }
    summary = stats.map(([k, v]) => `<div class="stat"><span class="stat-k">${k}</span><span class="stat-v">${esc(v)}</span></div>`).join("");
  }
  document.getElementById("day-summary").innerHTML = summary;

  const grid = document.getElementById("cards-grid");
  if (!games.length) {
    const opener = state.data.games.filter(g => g.type === "regular").map(g => g.date).sort()[0];
    const hint = opener && state.day < opener ? `The regular season tips off ${parseDay(opener).toLocaleDateString("en-US", { month: "long", day: "numeric" })}.`
      : state.team ? "This team is off — try another date or clear the filter."
      : "Pick a highlighted date on the calendar.";
    grid.innerHTML = `<div class="empty-state"><strong>No games on this date</strong><span>${hint}</span></div>`;
    return;
  }
  grid.innerHTML = games.map(renderCard).join("");
}

function defaultDay() {
  const days = [...new Set(visibleGames().map(g => g.date))].sort();
  const today = state.data.today;
  return days.find(d => d >= today) || days[days.length - 1] || today;
}

function selectDay(day) {
  state.day = day;
  state.month = monthKey(day);
  renderCalendar();
  renderDay();
}

function shiftMonth(delta) {
  const [y, m] = state.month.split("-").map(Number);
  const d = new Date(y, m - 1 + delta, 1);
  state.month = `${d.getFullYear()}-${pad(d.getMonth() + 1)}`;
  renderCalendar();
}

function setupCalendar() {
  const select = document.getElementById("team-filter");
  Object.entries(state.data.teams)
    .sort((x, y) => x[1].name.localeCompare(y[1].name))
    .forEach(([abbr, t]) => select.insertAdjacentHTML("beforeend", `<option value="${esc(abbr)}">${esc(t.name)}</option>`));

  select.addEventListener("change", () => {
    state.team = select.value;
    selectDay(defaultDay());
  });
  document.getElementById("cal-prev").addEventListener("click", () => shiftMonth(-1));
  document.getElementById("cal-next").addEventListener("click", () => shiftMonth(1));
  document.getElementById("cal-today").addEventListener("click", () => selectDay(state.data.today));
  selectDay(defaultDay());
}

// ── Standings ──
function renderStandings(standings) {
  const el = document.getElementById("standings");
  el.innerHTML = ["East", "West"].map(conf => `
    <div class="card conf-card">
      <div class="conf-head"><div class="conf-name">${conf}ern Conference</div><div class="conf-count">${standings[conf].length} teams</div></div>
      <div class="st-row st-headrow"><span>#</span><span>Team</span><span>W-L</span><span>Proj</span><span title="Roster rating: points per 100 possessions vs. an average team">Rating</span></div>
      ${standings[conf].map((r, i) => {
        const seed = i + 1, last = i === standings[conf].length - 1;
        const rating = state.data.teams[r.team].prior;
        const w = Math.min(Math.abs(rating) / 11 * 50, 50);
        const color = rating > 0 ? "var(--positive)" : rating < 0 ? "var(--negative)" : "oklch(0.65 0.01 260)";
        const zone = seed <= 6 ? "var(--positive)" : seed <= 10 ? "var(--warn)" : "transparent";
        const border = seed === 6 || seed === 10 ? "1px dashed var(--tick)" : last ? "0" : "1px solid var(--row-line)";
        return `
        <div class="st-row" style="border-bottom:${border};box-shadow:inset 3px 0 0 ${zone}">
          <span class="st-seed">${seed}</span>
          <button class="st-team" data-team="${esc(r.team)}">${tile(r.team, "tile-sm")}<span>${esc(r.name)}</span></button>
          <span class="st-wl">${r.w}-${r.l}</span>
          <span class="st-proj">${Math.round(r.proj_w)}-${Math.round(r.proj_l)}</span>
          <div class="st-rating">
            <div class="st-track"><div class="st-tick"></div>
              <div class="st-bar" style="left:${rating >= 0 ? 50 : 50 - w}%;width:${w}%;background:${color}"></div></div>
            <span class="st-val" style="color:${color}">${signed(rating)}</span>
          </div>
        </div>`;
      }).join("")}
    </div>`).join("");
  el.querySelectorAll(".st-team").forEach(b => b.addEventListener("click", () => {
    const select = document.getElementById("team-filter");
    select.value = b.dataset.team;
    select.dispatchEvent(new Event("change"));
    location.hash = "games";
  }));
}

// ── Accuracy ──
function renderAccuracy(acc, games) {
  const stats = document.getElementById("acc-stats");
  const recent = document.getElementById("acc-recent");
  if (!acc.games) {
    stats.innerHTML = "";
    recent.innerHTML = `<p class="empty-note">No games scored yet. Results appear here once the season tips off.</p>`;
    return;
  }
  stats.innerHTML = [["Pick record", `${acc.correct}-${acc.games - acc.correct}`], ["Accuracy", acc.pct + "%"],
    ["Avg pts off", acc.mae.toFixed(1)], ["Brier score", acc.brier.toFixed(3)]]
    .map(([k, v]) => `<div class="stat"><span class="stat-k">${k}</span><span class="stat-v">${v}</span></div>`).join("");

  const byId = Object.fromEntries(games.map(g => [g.id, g]));
  recent.innerHTML = `
    <div class="group-head"><strong>Recent results</strong><span>${acc.recent.length} games</span></div>
    ${acc.recent.map(r => {
      const g = byId[r.id];
      return `<div class="u-row">
        <span class="u-time">${esc(fmtShort(g.date))}</span>
        <span class="u-pill ${r.correct ? "k-good" : "k-injury"}">${r.correct ? "Called it" : "Missed"}</span>
        <div class="u-body">
          <div class="u-who">${tile(g.away, "tile-xs")}<strong>${esc(g.away)} ${g.result.away}</strong></div>
          <span class="u-detail">@</span>
          <div class="u-who">${tile(g.home, "tile-xs")}<strong>${esc(g.home)} ${g.result.home}</strong></div>
          <span class="u-detail">proj ${g.frozen.away.toFixed(0)}–${g.frozen.home.toFixed(0)}</span>
        </div>
      </div>`;
    }).join("")}`;
}

// ── Updates ──
function changeDetail(c) {
  const from = STATUS_LABEL[c.from] || c.from, to = STATUS_LABEL[c.to] || c.to;
  if (c.kind === "signed") return "Signed to roster";
  if (c.kind === "released") return "Waived";
  if (c.kind === "traded") return `Traded from ${c.from}`;
  if (c.kind === "cleared") return `Was ${String(from).toLowerCase()} · Available`;
  return [from && to ? `${from} → ${to}` : to, c.injury].filter(Boolean).join(" · ");
}

function renderChanges() {
  const changes = state.data.changes;
  const chips = [["all", "All", changes.length]].concat(Object.entries(CHANGE_TYPES)
    .map(([id, t]) => [id, t.chip, changes.filter(c => t.kinds.includes(c.kind)).length])
    .filter(([id, , n]) => n || id !== "traded"));
  const chipEl = document.getElementById("update-chips");
  chipEl.innerHTML = chips.map(([id, label, n]) =>
    `<button class="${state.updates === id ? "active" : ""}" data-filter="${id}">${label}<span>${n}</span></button>`).join("");
  chipEl.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.updates = b.dataset.filter;
    renderChanges();
  }));

  const shown = changes.filter(c => state.updates === "all" || CHANGE_TYPES[state.updates].kinds.includes(c.kind));
  const groups = [];
  shown.forEach(c => {
    const day = etDay(c.time);
    let g = groups.find(x => x.day === day);
    if (!g) groups.push(g = { day, items: [] });
    g.items.push(c);
  });
  document.getElementById("changes").innerHTML = groups.length ? groups.map(g => {
    const label = parseDay(g.day).toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" });
    return `<div>
      <div class="group-head"><strong>${g.day === state.data.today ? "Today · " : ""}${label}</strong>
        <span>${g.items.length} change${g.items.length === 1 ? "" : "s"}</span></div>
      ${g.items.map(c => {
        const [pill, cls] = CHANGE_PILL[c.kind] || [c.kind, "k-neutral"];
        return `<div class="u-row">
          <span class="u-time">${esc(fmtClock(c.time))}</span>
          <span class="u-pill ${cls}">${esc(pill)}</span>
          <div class="u-body">
            <div class="u-who">${tile(c.team, "tile-xs")}<strong>${esc(c.player)}</strong></div>
            <span class="u-detail">${esc(changeDetail(c))}</span>
          </div>
        </div>`;
      }).join("")}
    </div>`;
  }).join("") : `<p class="empty-note">No changes yet. The first snapshot is the baseline; changes show up from the next update.</p>`;
}

function renderInjuries(injuries) {
  const rows = Object.entries(injuries).flatMap(([team, list]) => list.map(i => ({ team, ...i })));
  document.getElementById("injury-report").innerHTML = rows.length ? `<div>
    <div class="group-head"><strong>Current injury report</strong><span>${rows.length} players</span></div>
    ${rows.map(i => `<div class="u-row">
      <span class="u-time">${esc(i.team)}</span>
      <span class="u-pill ${STATUS_PILL[i.status] || "k-neutral"}">${esc(STATUS_LABEL[i.status] || i.status)}</span>
      <div class="u-body">
        <div class="u-who">${tile(i.team, "tile-xs")}<strong>${esc(i.player)}</strong></div>
        <span class="u-detail">${esc(i.injury)}${i.return_date ? " · est. return " + esc(fmtShort(i.return_date)) : ""}</span>
      </div>
    </div>`).join("")}
  </div>` : "";
}

// ── Tabs ──
function showTab() {
  const tab = (location.hash || "#games").slice(1);
  const valid = ["games", "standings", "accuracy", "updates"].includes(tab) ? tab : "games";
  document.querySelectorAll(".tab-panel").forEach(p => { p.hidden = p.id !== "tab-" + valid; });
  document.querySelectorAll("nav a[data-tab]").forEach(a => a.classList.toggle("active", a.dataset.tab === valid));
}

async function init() {
  window.addEventListener("hashchange", showTab);
  showTab();
  try {
    state.data = await loadSeason();
    renderHeader(state.data);
    setupCalendar();
    renderStandings(state.data.standings);
    renderAccuracy(state.data.accuracy, state.data.games);
    renderChanges();
    renderInjuries(state.data.injuries);
  } catch (err) {
    console.error(err);
    document.getElementById("cards-grid").innerHTML =
      `<div class="empty-state"><strong>Could not load predictions</strong><span>${esc(err.message)}. Try refreshing.</span></div>`;
  }
}

init();
