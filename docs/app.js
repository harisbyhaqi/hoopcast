async function loadPredictions() {
  const res = await fetch("predictions.json");
  if (!res.ok) throw new Error("Failed to load predictions.json");
  return res.json();
}

function renderStats(stats) {
  document.getElementById("stat-mae").textContent  = stats.mae.toFixed(1);
  document.getElementById("stat-rmse").textContent = stats.rmse.toFixed(1);
  document.getElementById("stat-acc").textContent  = stats.win_accuracy + "%";
  document.getElementById("hero-acc").textContent  = stats.win_accuracy + "%";
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
      <div class="vs-divider">VS</div>
      <div class="team-block">
        <img class="team-logo" src="${m.team_b.logo}" alt="${m.team_b.abbr}" loading="lazy" onerror="this.style.display='none'">
        <span class="team-abbr">${m.team_b.abbr}</span>
        <span class="team-name">${m.team_b.name}</span>
      </div>
    </div>

    <div class="card-scores">
      <span class="score">${m.score_a.toFixed(1)}</span>
      <span class="score-sep">—</span>
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

    <div class="winner-badge">Predicted winner: ${winner.abbr}</div>
  `;
  return card;
}

async function init() {
  try {
    const data = await loadPredictions();
    renderStats(data.model_stats);

    const grid = document.getElementById("cards-grid");
    grid.innerHTML = "";
    data.matchups.forEach(m => grid.appendChild(renderCard(m)));

    if (data.generated) {
      document.getElementById("generated-note").textContent =
        `Predictions generated on ${data.generated} using 10 000 Monte Carlo simulations.`;
    }
  } catch (err) {
    console.error(err);
    document.getElementById("cards-grid").innerHTML =
      `<p style="color:#7a7f94">Could not load predictions.</p>`;
  }
}

init();
