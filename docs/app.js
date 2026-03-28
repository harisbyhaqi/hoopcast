async function loadPredictions() {
  const res = await fetch("predictions.json?v=" + Date.now());
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function renderStats(stats) {
  document.getElementById("stat-mae").textContent = stats.mae.toFixed(1) + " pts";
  document.getElementById("stat-acc").textContent = stats.win_accuracy + "%";
}

function formatDate(iso) {
  const d = new Date(iso + "T12:00:00");   // avoid TZ shift on date-only strings
  return d.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

function formatTimestamp(iso) {
  const d = new Date(iso);
  return d.toLocaleString("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
}

function renderCard(m) {
  const winner = m.win_prob_a >= m.win_prob_b ? m.team_a : m.team_b;

  const card = document.createElement("div");
  card.className = "match-card";

  card.innerHTML = `
    <div class="card-teams">
      <div class="team-block">
        <img class="team-logo" src="${m.team_a.logo}" alt="${m.team_a.abbr}" loading="lazy" onerror="this.style.display='none'">
        <span class="team-abbr">${m.team_a.abbr}</span>
        <span class="team-name">${m.team_a.name}</span>
      </div>
      <div class="vs-divider">@</div>
      <div class="team-block">
        <img class="team-logo" src="${m.team_b.logo}" alt="${m.team_b.abbr}" loading="lazy" onerror="this.style.display='none'">
        <span class="team-abbr">${m.team_b.abbr}</span>
        <span class="team-name">${m.team_b.name}</span>
      </div>
    </div>

    <div class="card-scores">
      <span class="score">${m.score_a.toFixed(1)}</span>
      <span class="score-sep">vs</span>
      <span class="score">${m.score_b.toFixed(1)}</span>
    </div>

    <div class="prob-bar-wrap">
      <div class="prob-bar">
        <div class="prob-bar-a" style="width: ${m.win_prob_a}%"></div>
        <div class="prob-bar-b" style="width: ${m.win_prob_b}%"></div>
      </div>
      <div class="prob-labels">
        <span class="pct-a">${m.win_prob_a}%</span>
        <span class="pct-b">${m.win_prob_b}%</span>
      </div>
    </div>

    <span class="pick-label">pick: <span>${winner.abbr}</span></span>
  `;
  return card;
}

async function init() {
  try {
    const data = await loadPredictions();
    renderStats(data.model_stats);

    // Section heading — show today's date
    if (data.date) {
      document.getElementById("matchups-date").textContent = formatDate(data.date);
    }

    const grid = document.getElementById("cards-grid");
    grid.innerHTML = "";

    if (!data.matchups || data.matchups.length === 0) {
      grid.innerHTML = `
        <div class="no-games">
          <span class="no-games-icon">🏀</span>
          <p>No games scheduled today.</p>
          <p class="no-games-sub">Check back tomorrow. Predictions update automatically each morning.</p>
        </div>`;
    } else {
      data.matchups.forEach(m => grid.appendChild(renderCard(m)));
    }

    // Last updated timestamp
    if (data.generated) {
      document.getElementById("generated-note").textContent =
        `updated ${formatTimestamp(data.generated)}`;
    }
  } catch (err) {
    console.error(err);
    document.getElementById("cards-grid").innerHTML =
      `<p style="color:#7a7f94;grid-column:1/-1">Could not load predictions (${err.message}). Try refreshing.</p>`;
  }
}

init();
