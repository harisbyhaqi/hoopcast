const state = { data: null, month: null, day: null, team: "" };

const STATUS_LABEL = { out: "Out", doubtful: "Doubtful", questionable: "Questionable", "day-to-day": "Day-to-day" };
const CHANGE_LABEL = {
  injury_new: "New injury", status_change: "Status change", cleared: "Cleared",
  signed: "Signed", released: "Released", traded: "Traded",
};

async function loadSeason() {
  const res = await fetch("season.json?v=" + Date.now());
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const pad = n => String(n).padStart(2, "0");
const isoDay = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const parseDay = iso => new Date(iso + "T12:00:00");   // avoid TZ shift on date-only strings
const monthKey = iso => iso.slice(0, 7);

function formatDate(iso) {
  return parseDay(iso).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}

function formatTip(isoUtc) {
  return new Date(isoUtc).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET";
}

function formatTimestamp(iso) {
  return new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
}

function visibleGames() {
  const t = state.team;
  return state.data.games.filter(g => !t || g.home === t || g.away === t);
}

// ── Hero ──
function renderHero(d) {
  document.getElementById("season-label").textContent = `${d.season} season`;
  const acc = d.accuracy;
  if (acc.games > 0) {
    document.getElementById("stat-acc").textContent = acc.pct + "%";
    document.getElementById("stat-acc-label").textContent = `picks right (${acc.correct}/${acc.games})`;
    document.getElementById("stat-mae").textContent = acc.mae.toFixed(1) + " pts";
  } else if (d.model_stats.win_accuracy != null) {
    document.getElementById("stat-acc").textContent = d.model_stats.win_accuracy + "%";
    document.getElementById("stat-acc-label").textContent = "win accuracy (back-test)";
    document.getElementById("stat-mae").textContent = d.model_stats.mae.toFixed(1) + " pts";
  }
  document.getElementById("stat-left").textContent = d.games.filter(g => g.type === "regular" && !g.result).length;
  document.getElementById("generated-note").textContent = `Last updated ${formatTimestamp(d.generated)}`;
}

// ── Calendar ──
function renderCalendar() {
  const [y, m] = state.month.split("-").map(Number);
  const first = new Date(y, m - 1, 1);
  document.getElementById("cal-month").textContent = first.toLocaleDateString("en-US", { month: "long", year: "numeric" });

  const byDay = {};
  visibleGames().forEach(g => (byDay[g.date] ??= []).push(g));

  const grid = document.getElementById("cal-grid");
  const cells = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map(w => `<div class="cal-dow">${w}</div>`);
  for (let i = 0; i < first.getDay(); i++) cells.push(`<div class="cal-cell empty"></div>`);

  const daysInMonth = new Date(y, m, 0).getDate();
  for (let day = 1; day <= daysInMonth; day++) {
    const iso = `${state.month}-${pad(day)}`;
    const games = byDay[iso] || [];
    const classes = ["cal-cell"];
    if (games.length) classes.push("has-games");
    if (iso === state.day) classes.push("selected");
    if (iso === state.data.today) classes.push("today");

    let detail = "";
    if (state.team && games.length) {
      const g = games[0];
      const opp = g.home === state.team ? g.away : "@" + g.home;
      let mark = "";
      if (g.result) {
        const won = (g.home === state.team) === (g.result.home > g.result.away);
        mark = `<span class="${won ? "w" : "l"}">${won ? "W" : "L"}</span>`;
      } else if (g.pred) {
        const p = g.home === state.team ? g.pred.p_home : 100 - g.pred.p_home;
        mark = `<span class="${p >= 50 ? "w" : "l"}">${Math.round(p)}%</span>`;
      }
      detail = `<span class="cal-detail">${esc(opp)} ${mark}</span>`;
    } else if (games.length) {
      detail = `<span class="cal-detail">${games.length} game${games.length > 1 ? "s" : ""}</span>`;
    }
    cells.push(`<button class="${classes.join(" ")}" data-day="${iso}" ${games.length ? "" : "disabled"}>
      <span class="cal-num">${day}</span>${detail}</button>`);
  }
  grid.innerHTML = cells.join("");
  grid.querySelectorAll("button.cal-cell").forEach(b => b.addEventListener("click", () => {
    state.day = b.dataset.day;
    renderCalendar();
    renderDay();
  }));
}

function injuryNote(list) {
  if (!list || !list.length) return "";
  return list.map(i => `${esc(i.player)} (${esc(STATUS_LABEL[i.status] || i.status)}, −${i.pts_lost})`).join(", ");
}

function renderCard(g) {
  const teams = state.data.teams;
  const a = { abbr: g.away, ...teams[g.away] }, b = { abbr: g.home, ...teams[g.home] };
  const final = !!g.result;
  const shown = final ? g.result : g.pred;
  const probs = g.pred || g.frozen;
  const pHome = probs ? probs.p_home : null;

  let verdict = "";
  if (final && g.frozen) {
    const right = (g.frozen.p_home >= 50) === (g.result.home > g.result.away);
    verdict = `<span class="verdict ${right ? "right" : "wrong"}">${right ? "✓ called it" : "✗ missed"}</span>`;
  }

  const card = document.createElement("div");
  card.className = "match-card" + (final ? " final" : "");
  card.innerHTML = `
    <div class="card-meta">
      <span>${final ? "Final" : esc(formatTip(g.tip))}${g.type === "playoffs" ? " · Playoffs" : ""}</span>
      ${verdict}
    </div>
    <div class="card-teams">
      <div class="team-block">
        <img class="team-logo" src="${esc(a.logo)}" alt="${esc(a.abbr)}" loading="lazy" onerror="this.style.display='none'">
        <span class="team-abbr">${esc(a.abbr)}</span>
        <span class="team-name">${esc(a.name)}</span>
      </div>
      <div class="vs-divider">@</div>
      <div class="team-block">
        <img class="team-logo" src="${esc(b.logo)}" alt="${esc(b.abbr)}" loading="lazy" onerror="this.style.display='none'">
        <span class="team-abbr">${esc(b.abbr)}</span>
        <span class="team-name">${esc(b.name)}</span>
      </div>
    </div>
    ${shown ? `
    <div class="card-scores">
      <span class="score">${final ? shown.away : shown.away.toFixed(1)}</span>
      <span class="score-sep">${final ? "final" : "proj"}</span>
      <span class="score">${final ? shown.home : shown.home.toFixed(1)}</span>
    </div>` : ""}
    ${final && g.frozen ? `<p class="card-sub">We said ${g.frozen.away.toFixed(1)} – ${g.frozen.home.toFixed(1)}</p>` : ""}
    ${pHome != null ? `
    <div class="prob-bar-wrap">
      <div class="prob-bar">
        <div class="prob-bar-a" style="width: ${100 - pHome}%"></div>
        <div class="prob-bar-b" style="width: ${pHome}%"></div>
      </div>
      <div class="prob-labels">
        <span class="pct-a">${(100 - pHome).toFixed(1)}%</span>
        <span class="pct-b">${pHome.toFixed(1)}%</span>
      </div>
    </div>
    <span class="pick-label">pick: <span>${pHome >= 50 ? esc(b.abbr) : esc(a.abbr)}</span></span>` : ""}
    ${g.injuries ? `<p class="card-injuries">${[injuryNote(g.injuries.away), injuryNote(g.injuries.home)].filter(Boolean).join(" · ")}</p>` : ""}
  `;
  return card;
}

function renderDay() {
  const games = visibleGames().filter(g => g.date === state.day);
  document.getElementById("matchups-date").textContent = state.day ? formatDate(state.day) : "No games";
  const grid = document.getElementById("cards-grid");
  grid.innerHTML = "";
  if (!games.length) {
    grid.innerHTML = `
      <div class="no-games">
        <span class="no-games-icon">🏀</span>
        <p>No games on this day.</p>
        <p class="no-games-sub">Pick a highlighted day on the calendar.</p>
      </div>`;
    return;
  }
  games.forEach(g => grid.appendChild(renderCard(g)));
}

function defaultDay() {
  const days = [...new Set(visibleGames().map(g => g.date))].sort();
  const today = state.data.today;
  return days.find(d => d >= today) || days[days.length - 1] || today;
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
    state.day = defaultDay();
    state.month = monthKey(state.day);
    renderCalendar();
    renderDay();
  });
  document.getElementById("cal-prev").addEventListener("click", () => shiftMonth(-1));
  document.getElementById("cal-next").addEventListener("click", () => shiftMonth(1));
  document.getElementById("cal-today").addEventListener("click", () => {
    state.day = defaultDay();
    state.month = monthKey(state.day);
    renderCalendar();
    renderDay();
  });

  state.day = defaultDay();
  state.month = monthKey(state.day);
  renderCalendar();
  renderDay();
}

// ── Standings ──
function renderStandings(standings) {
  const el = document.getElementById("standings");
  el.innerHTML = ["East", "West"].map(conf => `
    <div class="standings-table">
      <h3>${conf}ern Conference</h3>
      <table>
        <thead><tr><th>#</th><th>Team</th><th>W-L</th><th>Proj.</th></tr></thead>
        <tbody>
          ${standings[conf].map((r, i) => `
            <tr class="${i === 5 ? "cut-playoff" : i === 9 ? "cut-playin" : ""}">
              <td>${i + 1}</td>
              <td><button class="link-btn" data-team="${esc(r.team)}">${esc(r.name)}</button></td>
              <td>${r.w}-${r.l}</td>
              <td>${Math.round(r.proj_w)}-${Math.round(r.proj_l)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`).join("");
  el.querySelectorAll(".link-btn").forEach(b => b.addEventListener("click", () => {
    const select = document.getElementById("team-filter");
    select.value = b.dataset.team;
    select.dispatchEvent(new Event("change"));
    location.hash = "calendar";
  }));
}

// ── Accuracy ──
function renderAccuracy(acc, games) {
  const stats = document.getElementById("acc-stats");
  if (!acc.games) {
    stats.innerHTML = `<p class="empty-note">No games scored yet. Results appear here once the season tips off.</p>`;
    document.getElementById("acc-recent").innerHTML = "";
    return;
  }
  stats.innerHTML = `
    <div class="stat-block"><span class="stat-val">${acc.correct}-${acc.games - acc.correct}</span><span class="stat-label">pick record</span></div>
    <div class="stat-block"><span class="stat-val">${acc.pct}%</span><span class="stat-label">accuracy</span></div>
    <div class="stat-block"><span class="stat-val">${acc.mae.toFixed(1)}</span><span class="stat-label">avg pts off</span></div>
    <div class="stat-block"><span class="stat-val">${acc.brier.toFixed(3)}</span><span class="stat-label">Brier score</span></div>`;

  const byId = Object.fromEntries(games.map(g => [g.id, g]));
  document.getElementById("acc-recent").innerHTML = acc.recent.map(r => {
    const g = byId[r.id];
    return `<div class="result-row">
      <span class="verdict ${r.correct ? "right" : "wrong"}">${r.correct ? "✓" : "✗"}</span>
      <span class="result-date">${esc(parseDay(g.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }))}</span>
      <span>${esc(g.away)} ${g.result.away} @ ${esc(g.home)} ${g.result.home}</span>
      <span class="muted">proj ${g.frozen.away.toFixed(0)}–${g.frozen.home.toFixed(0)}</span>
    </div>`;
  }).join("");
}

// ── Updates ──
function renderUpdates(changes, injuries) {
  const list = document.getElementById("changes");
  list.innerHTML = changes.length ? changes.map(c => {
    const move = c.from || c.to ? `${esc(STATUS_LABEL[c.from] || c.from || "")}${c.from && c.to ? " → " : ""}${esc(STATUS_LABEL[c.to] || c.to || "")}` : "";
    return `<div class="change-row">
      <span class="result-date">${esc(formatTimestamp(c.time))}</span>
      <span class="badge badge-${esc(c.kind)}">${esc(CHANGE_LABEL[c.kind] || c.kind)}</span>
      <span><strong>${esc(c.player)}</strong> <span class="muted">${esc(c.team)}</span></span>
      <span class="muted">${move}${c.injury ? " · " + esc(c.injury) : ""}</span>
    </div>`;
  }).join("") : `<p class="empty-note">No changes yet. The first snapshot is the baseline; changes show up from the next update.</p>`;

  document.getElementById("injury-report").innerHTML = Object.entries(injuries).map(([team, rows]) => `
    <details class="injury-team">
      <summary>${esc(state.data.teams[team]?.name || team)} <span class="muted">(${rows.length})</span></summary>
      ${rows.map(i => `<div class="injury-row">
        <span>${esc(i.player)}</span>
        <span class="badge badge-${i.status === "out" ? "injury_new" : "status_change"}">${esc(STATUS_LABEL[i.status] || i.status)}</span>
        <span class="muted">${esc(i.injury)}${i.return_date ? " · est. return " + esc(i.return_date) : ""}</span>
      </div>`).join("")}
    </details>`).join("");
}

// ── Tabs ──
function showTab() {
  const tab = (location.hash || "#calendar").slice(1);
  const valid = ["calendar", "standings", "accuracy", "updates"].includes(tab) ? tab : "calendar";
  document.querySelectorAll(".tab-panel").forEach(p => { p.hidden = p.id !== "tab-" + valid; });
  document.querySelectorAll("nav a[data-tab]").forEach(a => a.classList.toggle("active", a.dataset.tab === valid));
}

async function init() {
  window.addEventListener("hashchange", showTab);
  showTab();
  try {
    state.data = await loadSeason();
    renderHero(state.data);
    setupCalendar();
    renderStandings(state.data.standings);
    renderAccuracy(state.data.accuracy, state.data.games);
    renderUpdates(state.data.changes, state.data.injuries);
  } catch (err) {
    console.error(err);
    document.getElementById("cards-grid").innerHTML =
      `<p style="color:#7a7f94;grid-column:1/-1">Could not load predictions (${esc(err.message)}). Try refreshing.</p>`;
  }
}

init();
