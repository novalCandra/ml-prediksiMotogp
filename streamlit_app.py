import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import os

st.set_page_config(
    page_title="Prediksi Motogp",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- fonts & base ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: "Inter", sans-serif; }

/* ---- hide default header/footer ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- page background ---- */
.stApp { background: #0d0f14; color: #e8eaf0; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] * { color: #c8cad4 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 6px 0; }

/* ---- hero banner ---- */
.hero {
    background: linear-gradient(135deg, #c8102e 0%, #8b0000 50%, #1a0a0a 100%);
    border-radius: 16px;
    padding: 48px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🏍️";
    position: absolute;
    right: 40px; top: 20px;
    font-size: 120px;
    opacity: 0.12;
}
.hero h1 { font-size: 2.6rem; font-weight: 900; color: #fff; margin: 0 0 8px; }
.hero p  { font-size: 1.05rem; color: rgba(255,255,255,0.75); margin: 0; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #fff;
    margin-top: 14px;
    margin-right: 6px;
}

/* ---- section title ---- */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e8eaf0;
    border-left: 4px solid #c8102e;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* ---- podium cards ---- */
.podium-card {
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.podium-card.gold   { background: linear-gradient(145deg,#b8860b,#ffd700); color:#1a1200; }
.podium-card.silver { background: linear-gradient(145deg,#5a5a5a,#c0c0c0); color:#111; }
.podium-card.bronze { background: linear-gradient(145deg,#7c4a1e,#cd7f32); color:#1a0a00; }
.podium-card .medal { font-size: 2.4rem; margin-bottom: 6px; }
.podium-card .name  { font-size: 1.2rem; font-weight: 800; margin: 4px 0; }
.podium-card .prob  { font-size: 2rem; font-weight: 900; margin: 8px 0; }
.podium-card .divider { border: 1px solid rgba(0,0,0,0.2); margin: 10px 0; }
.podium-card .stat  { font-size: 0.82rem; margin: 3px 0; opacity: 0.85; }
.podium-card .team-tag {
    display: inline-block;
    background: rgba(0,0,0,0.18);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-top: 8px;
}

/* ---- stat tiles ---- */
.stat-tile {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
}
.stat-tile .label { font-size: 0.75rem; color: #7a7f96; text-transform: uppercase; letter-spacing: .06em; }
.stat-tile .value { font-size: 1.7rem; font-weight: 800; color: #e8eaf0; margin: 4px 0 2px; }
.stat-tile .delta { font-size: 0.78rem; color: #4caf7d; }
.stat-tile .delta.warn { color: #f0a500; }

/* ---- rider profile card ---- */
.profile-card {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 14px;
    padding: 24px;
}
.profile-card h3 { font-size: 1.4rem; font-weight: 800; color: #fff; margin: 0 0 4px; }
.profile-card .subtitle { color: #7a7f96; font-size: 0.85rem; margin-bottom: 16px; }
.profile-card .kpi-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.profile-card .kpi {
    background: #0d0f14;
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1; min-width: 80px;
    text-align: center;
}
.profile-card .kpi .kv { font-size: 1.3rem; font-weight: 800; color: #c8102e; }
.profile-card .kpi .kl { font-size: 0.7rem; color: #7a7f96; text-transform: uppercase; }

/* ---- strength / challenge pills ---- */
.pill-green {
    display: inline-block;
    background: rgba(76,175,125,0.15);
    border: 1px solid rgba(76,175,125,0.35);
    color: #4caf7d;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 3px 3px 3px 0;
}
.pill-red {
    display: inline-block;
    background: rgba(200,16,46,0.15);
    border: 1px solid rgba(200,16,46,0.35);
    color: #e05070;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 3px 3px 3px 0;
}

/* ---- data table ---- */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ---- viz card ---- */
.viz-card {
    background: #1a1d27;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.viz-card h4 { font-size: 0.9rem; color: #7a7f96; text-transform: uppercase;
               letter-spacing:.06em; margin: 0 0 12px; }

/* ---- footer ---- */
.footer {
    text-align: center;
    color: #3a3f52;
    font-size: 0.78rem;
    padding: 24px 0 8px;
    border-top: 1px solid #1f2330;
    margin-top: 40px;
}

/* ---- progress bar ---- */
.prob-bar-wrap { background:#1a1d27; border-radius:6px; height:8px; margin-top:4px; }
.prob-bar { background: linear-gradient(90deg,#c8102e,#ff6b6b); border-radius:6px; height:8px; }

/* ---- info box ---- */
.info-box {
    background: #1a1d27;
    border-left: 4px solid #c8102e;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #c8cad4;
}

/* ---- manufacturer badge ---- */
.mfr { display:inline-block; border-radius:6px; padding:2px 10px;
       font-size:0.75rem; font-weight:700; margin-left:6px; }
.mfr-ducati  { background:#e8002d22; color:#e8002d; border:1px solid #e8002d44; }
.mfr-ktm     { background:#ff690022; color:#ff6900; border:1px solid #ff690044; }
.mfr-yamaha  { background:#0033a022; color:#4488dd; border:1px solid #0033a044; }
.mfr-honda   { background:#cc000022; color:#cc6666; border:1px solid #cc000044; }
.mfr-aprilia { background:#00529b22; color:#5599cc; border:1px solid #00529b44; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/championship_predictions_2025.csv")

@st.cache_data
def load_future_data():
    path = "data/future_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()
fdf_future = load_future_data()

MFR_CLASS = {
    "Ducati": "mfr-ducati", "KTM": "mfr-ktm", "KTM RC16": "mfr-ktm",
    "Yamaha": "mfr-yamaha", "Honda": "mfr-honda", "Aprilia": "mfr-aprilia",
}

def mfr_badge(name):
    key = name.split()[0] if name else ""
    cls = MFR_CLASS.get(key, "mfr-ducati")
    return f"<span class='mfr {cls}'>{name}</span>"

FLAG = {
    "Spain":"🇪🇸","Italy":"🇮🇹","France":"🇫🇷","Australia":"🇦🇺",
    "South Africa":"🇿🇦","Japan":"🇯🇵","Portugal":"🇵🇹","Thailand":"🇹🇭",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏍️ MotoGP Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏆 Championship Forecast",
         "📕 Future Forecast 2026–2030",
         "📊 Rider Analysis",
         "📈 Visualizations",
         "📋 Full Grid",
         "ℹ️ Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
<div style='background:#0d0f14;border-radius:10px;padding:14px;font-size:0.8rem;color:#7a7f96;'>
<div style='color:#c8102e;font-weight:700;margin-bottom:8px;'>MODEL SUMMARY</div>
<div>Algorithm: Random Forest</div>
<div>Accuracy: <b style='color:#4caf7d'>81.82%</b></div>
<div>ROC-AUC: <b style='color:#4caf7d'>0.90</b></div>
<div>Training: 2022–2024</div>
<div>Features: 12 engineered</div>
</div>
""", unsafe_allow_html=True)



# ── Helpers ───────────────────────────────────────────────────────────────────
def prob_bar(pct):
    """Render a thin red progress bar."""
    return f"""
    <div class='prob-bar-wrap'>
      <div class='prob-bar' style='width:{pct*100:.1f}%'></div>
    </div>"""

def dark_chart(fig):
    """Apply dark theme to a matplotlib figure."""
    fig.patch.set_facecolor("#1a1d27")
    for ax in fig.get_axes():
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#7a7f96")
        ax.xaxis.label.set_color("#7a7f96")
        ax.yaxis.label.set_color("#7a7f96")
        ax.title.set_color("#e8eaf0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#252836")
    return fig

# ── PAGE 1 — Championship Forecast ───────────────────────────────────────────
if page == "🏆 Championship Forecast":

    # Hero banner
    st.markdown("""
    <div class='hero'>
      <h1>2025 MotoGP Championship Forecast</h1>
      <p>Machine learning predictions based on 2013–2024 historical performance data</p>
      <span class='badge'>Random Forest</span>
      <span class='badge'>81.82% Accuracy</span>
      <span class='badge'>ROC-AUC 0.90</span>
      <span class='badge'>22 Riders</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Podium cards ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Predicted Podium</div>", unsafe_allow_html=True)

    top3 = df.head(3)
    styles = [
        ("gold",   "🥇", "1st"),
        ("silver", "🥈", "2nd"),
        ("bronze", "🥉", "3rd"),
    ]

    cols = st.columns(3)
    for col, (style, medal, rank), (_, row) in zip(cols, styles, top3.iterrows()):
        flag = FLAG.get(row["Country"], "")
        with col:
            st.markdown(f"""
            <div class='podium-card {style}'>
              <div class='medal'>{medal}</div>
              <div style='font-size:0.75rem;font-weight:600;opacity:.7;'>{rank} PLACE</div>
              <div class='name'>{flag} {row['Rider']}</div>
              <div class='prob'>{row['Championship Probability']:.0%}</div>
              {prob_bar(row['Championship Probability'])}
              <div class='divider'></div>
              <div class='stat'>🏁 Points: <b>{int(row['Points'])}</b></div>
              <div class='stat'>🏆 Wins: <b>{int(row['Wins'])}</b></div>
              <div class='stat'>🎖️ Podiums: <b>{int(row['Podiums'])}</b></div>
              <div class='stat'>⚡ Poles: <b>{int(row['Poles'])}</b></div>
              <div class='stat'>📍 PPR: <b>{row['PPR']:.1f}</b></div>
              <div class='team-tag'>{row['Team']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick-stat tiles ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Season Snapshot</div>", unsafe_allow_html=True)

    leader = df.iloc[0]
    t1, t2, t3, t4, t5 = st.columns(5)
    tiles = [
        (t1, "Championship Leader", leader["Rider"].split()[-1], "2025 Season"),
        (t2, "Leader Probability",  f"{leader['Championship Probability']:.0%}", "Random Forest"),
        (t3, "Points Leader",       str(int(df['Points'].max())), "Highest so far"),
        (t4, "Riders in Grid",      str(len(df)), "2025 Season"),
        (t5, "Ducati Riders",       str((df['Motorcycle'].str.contains('Ducati')).sum()), "Most represented"),
    ]
    for col, label, val, delta in tiles:
        with col:
            st.markdown(f"""
            <div class='stat-tile'>
              <div class='label'>{label}</div>
              <div class='value'>{val}</div>
              <div class='delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability bar chart ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Championship Probability — Top 10</div>", unsafe_allow_html=True)

    top10 = df[df["Championship Probability"] > 0].head(10).sort_values("Championship Probability")
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ["#c8102e" if i == len(top10) - 1 else
                  "#e05070" if i == len(top10) - 2 else
                  "#8b3a4a" if i == len(top10) - 3 else "#3a2030"
                  for i in range(len(top10))]
    bars = ax.barh(top10["Rider"], top10["Championship Probability"] * 100,
                   color=bar_colors, height=0.6)
    for bar, val in zip(bars, top10["Championship Probability"] * 100):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e8eaf0", fontsize=9, fontweight="bold")
    ax.set_xlabel("Championship Probability (%)", color="#7a7f96")
    ax.set_xlim(0, top10["Championship Probability"].max() * 120)
    ax.set_title("Predicted Championship Probability", color="#e8eaf0", fontsize=12, fontweight="bold")
    dark_chart(fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Key factors ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Why Marc Márquez Leads</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='profile-card'>
          <h3>Key Strengths</h3>
          <div class='subtitle'>Factors driving the prediction</div>
          <span class='pill-green'>✓ Factory Ducati upgrade</span>
          <span class='pill-green'>✓ 6× MotoGP champion</span>
          <span class='pill-green'>✓ 100% podium rate (2025)</span>
          <span class='pill-green'>✓ Pole + win in race 1</span>
          <span class='pill-green'>✓ Most competitive bike</span>
          <span class='pill-green'>✓ 12+ years MotoGP experience</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='profile-card'>
          <h3>Risk Factors</h3>
          <div class='subtitle'>Limitations the model cannot capture</div>
          <span class='pill-red'>⚠ Injury history (shoulder)</span>
          <span class='pill-red'>⚠ Intra-team rivalry (Bagnaia)</span>
          <span class='pill-red'>⚠ Only 1 race of 2025 data</span>
          <span class='pill-red'>⚠ Mid-season bike changes</span>
          <span class='pill-red'>⚠ Unpredictable crashes</span>
          <span class='pill-red'>⚠ Regulatory changes</span>
        </div>""", unsafe_allow_html=True)


# ── PAGE 2 — Rider Analysis ───────────────────────────────────────────────────
elif page == "📊 Rider Analysis":

    st.markdown("""
    <div class='hero' style='padding:32px 40px;'>
      <h1 style='font-size:2rem;'>Rider Analysis</h1>
      <p>Compare performance metrics across the 2025 grid</p>
    </div>""", unsafe_allow_html=True)

    # Rider selector
    st.markdown("<div class='section-title'>Select a Rider</div>", unsafe_allow_html=True)
    rider_name = st.selectbox("", df["Rider"].tolist(), label_visibility="collapsed")
    row = df[df["Rider"] == rider_name].iloc[0]
    flag = FLAG.get(row["Country"], "")

    # Profile card
    prob_pct = row["Championship Probability"] * 100
    st.markdown(f"""
    <div class='profile-card' style='margin-bottom:20px;'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;'>
        <div>
          <h3>{flag} {row['Rider']}</h3>
          <div class='subtitle'>{row['Team']} &nbsp;·&nbsp; {mfr_badge(row['Motorcycle'])}</div>
          <div style='margin-top:10px;'>
            <span style='font-size:2.2rem;font-weight:900;color:#c8102e;'>{prob_pct:.1f}%</span>
            <span style='color:#7a7f96;font-size:0.85rem;margin-left:8px;'>championship probability</span>
          </div>
          {prob_bar(row['Championship Probability'])}
        </div>
        <div style='text-align:right;'>
          <div style='font-size:0.75rem;color:#7a7f96;text-transform:uppercase;'>Country</div>
          <div style='font-size:1rem;font-weight:600;color:#e8eaf0;'>{flag} {row['Country']}</div>
        </div>
      </div>
      <div class='kpi-row'>
        <div class='kpi'><div class='kv'>{int(row['Points'])}</div><div class='kl'>Points</div></div>
        <div class='kpi'><div class='kv'>{int(row['Wins'])}</div><div class='kl'>Wins</div></div>
        <div class='kpi'><div class='kv'>{int(row['Podiums'])}</div><div class='kl'>Podiums</div></div>
        <div class='kpi'><div class='kv'>{int(row['Poles'])}</div><div class='kl'>Poles</div></div>
        <div class='kpi'><div class='kv'>{int(row['Fastest Laps'])}</div><div class='kl'>Fast Laps</div></div>
        <div class='kpi'><div class='kv'>{row['PPR']:.1f}</div><div class='kl'>PPR</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Radar chart vs grid average
    st.markdown("<div class='section-title'>Performance Radar vs Grid Average</div>", unsafe_allow_html=True)

    metrics = ["Points", "Wins", "Podiums", "Poles", "Fastest Laps", "PPR"]
    grid_max = df[metrics].max()
    rider_vals = [row[m] / grid_max[m] if grid_max[m] > 0 else 0 for m in metrics]
    avg_vals   = [(df[m].mean() / grid_max[m]) if grid_max[m] > 0 else 0 for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    rider_vals += rider_vals[:1]; avg_vals += avg_vals[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#1a1d27")
    ax.plot(angles, rider_vals, color="#c8102e", linewidth=2)
    ax.fill(angles, rider_vals, color="#c8102e", alpha=0.25)
    ax.plot(angles, avg_vals, color="#4a90d9", linewidth=1.5, linestyle="--")
    ax.fill(angles, avg_vals, color="#4a90d9", alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color="#c8cad4", fontsize=9)
    ax.set_yticklabels([])
    ax.grid(color="#252836", linewidth=0.8)
    ax.spines["polar"].set_color("#252836")
    legend = [mpatches.Patch(color="#c8102e", label=rider_name),
              mpatches.Patch(color="#4a90d9", label="Grid Average")]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=8, framealpha=0, labelcolor="#c8cad4")
    plt.tight_layout()

    col_radar, col_compare = st.columns([1, 1])
    with col_radar:
        st.pyplot(fig)
        plt.close()

    # Head-to-head bar vs top 3
    with col_compare:
        st.markdown("<div style='color:#7a7f96;font-size:0.8rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;'>Points vs Top 3</div>", unsafe_allow_html=True)
        compare_riders = df.head(3)["Rider"].tolist()
        if rider_name not in compare_riders:
            compare_riders.append(rider_name)
        compare_df = df[df["Rider"].isin(compare_riders)].copy()

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bar_c = ["#c8102e" if r == rider_name else "#3a2030" for r in compare_df["Rider"]]
        ax2.bar(compare_df["Rider"], compare_df["Points"], color=bar_c, width=0.5)
        ax2.set_ylabel("Points", color="#7a7f96")
        ax2.set_title("Points Comparison", color="#e8eaf0", fontsize=11)
        plt.xticks(rotation=15, ha="right", fontsize=8)
        dark_chart(fig2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Metric breakdown table
    st.markdown("<div class='section-title'>Full Metric Breakdown</div>", unsafe_allow_html=True)
    breakdown = pd.DataFrame({
        "Metric": metrics,
        "Rider Value": [row[m] for m in metrics],
        "Grid Average": [round(df[m].mean(), 2) for m in metrics],
        "Grid Max": [df[m].max() for m in metrics],
        "Percentile": [
            f"{int((df[m] <= row[m]).mean() * 100)}th"
            for m in metrics
        ]
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True)


# ── PAGE 3 — Visualizations ───────────────────────────────────────────────────
elif page == "📈 Visualizations":

    st.markdown("""
    <div class='hero' style='padding:32px 40px;'>
      <h1 style='font-size:2rem;'>Visualizations</h1>
      <p>All 9 pre-generated analysis charts</p>
    </div>""", unsafe_allow_html=True)

    viz_files = {
        "Championship Probability":       "data/championship_probability.png",
        "Top 3 Detailed Comparison":      "data/top3_detailed_comparison.png",
        "Feature Importance":             "data/feature_importance.png",
        "Correlation Analysis":           "data/correlation_analysis.png",
        "Performance Heatmap":            "data/performance_heatmap.png",
        "Points Per Race vs Probability": "data/ppr_vs_probability.png",
        "Team Distribution":              "data/team_distribution.png",
        "Motorcycle Distribution":        "data/motorcycle_distribution.png",
        "Country Distribution":           "data/country_distribution.png",
    }

    # Featured selector
    st.markdown("<div class='section-title'>Featured Chart</div>", unsafe_allow_html=True)
    selected = st.selectbox("", list(viz_files.keys()), label_visibility="collapsed")
    path = viz_files[selected]
    if os.path.exists(path):
        st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
        st.image(Image.open(path), use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Chart not found: {path}. Run `create_visualizations.py` first.")

    # Grid of all charts
    st.markdown("<div class='section-title'>All Charts</div>", unsafe_allow_html=True)
    items = list(viz_files.items())
    for i in range(0, len(items), 2):
        c1, c2 = st.columns(2)
        for col, (name, fpath) in zip([c1, c2], items[i:i+2]):
            if os.path.exists(fpath):
                with col:
                    st.markdown(f"<div class='viz-card'><h4>{name}</h4>", unsafe_allow_html=True)
                    st.image(Image.open(fpath), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE 4 — Full Grid ────────────────────────────────────────────────────────
elif page == "📋 Full Grid":

    st.markdown("""
    <div class='hero' style='padding:32px 40px;'>
      <h1 style='font-size:2rem;'>2025 Full Grid Rankings</h1>
      <p>All 22 riders ranked by predicted championship probability</p>
    </div>""", unsafe_allow_html=True)

    # Filters
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_prob = st.slider("Min probability (%)", 0, 100, 0)
    with fc2:
        moto_opts = sorted(df["Motorcycle"].unique())
        sel_moto = st.multiselect("Motorcycle", moto_opts, default=moto_opts)
    with fc3:
        country_opts = sorted(df["Country"].unique())
        sel_country = st.multiselect("Country", country_opts, default=country_opts)

    fdf = df[
        (df["Championship Probability"] * 100 >= min_prob) &
        (df["Motorcycle"].isin(sel_moto)) &
        (df["Country"].isin(sel_country))
    ].copy().reset_index(drop=True)
    fdf.insert(0, "Rank", fdf.index + 1)
    fdf["Prob %"] = (fdf["Championship Probability"] * 100).round(1)
    fdf["Flag"] = fdf["Country"].map(FLAG).fillna("")
    fdf["Rider"] = fdf["Flag"] + " " + fdf["Rider"]

    # Summary tiles
    st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val, delta in [
        (s1, "Riders shown",    len(fdf),                          "after filters"),
        (s2, "Avg probability", f"{fdf['Championship Probability'].mean():.1%}", "across grid"),
        (s3, "Max points",
        int(fdf["Points"].max()) if not fdf.empty and not fdf["Points"].isna().all() else 0,
        "2025 season"),
        (s4, "Total podiums",   int(fdf["Podiums"].sum()),          "combined"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-tile'>
              <div class='label'>{label}</div>
              <div class='value'>{val}</div>
              <div class='delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    st.markdown("<div class='section-title'>Rankings Table</div>", unsafe_allow_html=True)
    display_cols = ["Rank", "Rider", "Team", "Motorcycle", "Points", "Wins", "Podiums", "Poles", "Fastest Laps", "PPR", "Prob %"]
    st.dataframe(
        fdf[display_cols].style
            .background_gradient(subset=["Prob %"], cmap="Reds")
            .background_gradient(subset=["Points"], cmap="Blues")
            .format({"PPR": "{:.1f}", "Prob %": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # Download
    st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
    csv = fdf[display_cols].to_csv(index=False)
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="motogp_2025_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ── PAGE 5 — Model Info ───────────────────────────────────────────────────────
elif page == "ℹ️ Model Info":

    st.markdown("""
    <div class='hero' style='padding:32px 40px;'>
      <h1 style='font-size:2rem;'>Model Information</h1>
      <p>How the championship predictor was built and evaluated</p>
    </div>""", unsafe_allow_html=True)

    # Model performance metrics
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, delta, warn in [
        (m1, "Accuracy",    "81.82%", "Test set",       False),
        (m2, "ROC-AUC",     "0.90",   "Excellent",      False),
        (m3, "Precision",   "90%",    "Non-champions",  False),
        (m4, "Training set","55",     "2022–2024 rows", True),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-tile'>
              <div class='label'>{label}</div>
              <div class='value'>{val}</div>
              <div class='delta {"warn" if warn else ""}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
    features = ["Podiums", "Points", "Pole Freq.", "Podium Freq.", "Poles",
                "PPR", "Win Rate", "Wins", "FL Freq.", "Fastest Laps", "Races", "Consistency"]
    importances = [0.258, 0.169, 0.134, 0.123, 0.108, 0.085, 0.064, 0.057, 0.002, 0.001, 0.0, 0.0]
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ["#c8102e" if v > 0.15 else "#8b3a4a" if v > 0.08 else "#3a2030" for v in fi_df["Importance"]]
    ax.barh(fi_df["Feature"], fi_df["Importance"] * 100, color=bar_colors, height=0.6)
    for i, (_, r) in enumerate(fi_df.iterrows()):
        if r["Importance"] > 0:
            ax.text(r["Importance"] * 100 + 0.3, i, f"{r['Importance']*100:.1f}%",
                    va="center", color="#e8eaf0", fontsize=8.5)
    ax.set_xlabel("Importance (%)", color="#7a7f96")
    ax.set_title("Random Forest Feature Importance", color="#e8eaf0", fontsize=12, fontweight="bold")
    dark_chart(fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation + methodology side by side
    st.markdown("<div class='section-title'>Correlation with Championship</div>", unsafe_allow_html=True)
    corr_data = {
        "Metric": ["Podiums", "Poles", "Wins", "Podium Freq.", "Fastest Laps",
                   "Pole Freq.", "Win Rate", "Points", "PPR", "FL Freq.", "Races", "Avg Finish"],
        "Correlation": [0.725, 0.718, 0.705, 0.622, 0.617, 0.590, 0.581, 0.542, 0.456, 0.411, 0.154, -0.444],
    }
    corr_df = pd.DataFrame(corr_data).sort_values("Correlation")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bar_c2 = ["#c8102e" if v > 0 else "#4a90d9" for v in corr_df["Correlation"]]
    ax2.barh(corr_df["Metric"], corr_df["Correlation"], color=bar_c2, height=0.6)
    ax2.axvline(0, color="#7a7f96", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Pearson Correlation", color="#7a7f96")
    ax2.set_title("Feature Correlation with Championship Win", color="#e8eaf0", fontsize=12, fontweight="bold")
    dark_chart(fig2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Methodology cards
    st.markdown("<div class='section-title'>Methodology</div>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div class='profile-card'>
          <h3>Algorithm</h3>
          <div class='subtitle'>Random Forest Classifier</div>
          <div class='info-box'>
            100 decision trees · max depth 10 · 12 engineered features ·
            stratified 80/20 train-test split · StandardScaler normalisation
          </div>
          <div style='margin-top:14px;'>
            <div style='color:#7a7f96;font-size:0.78rem;text-transform:uppercase;margin-bottom:6px;'>Why Random Forest?</div>
            <span class='pill-green'>Handles non-linear patterns</span>
            <span class='pill-green'>Built-in feature importance</span>
            <span class='pill-green'>Robust to class imbalance</span>
            <span class='pill-green'>Calibrated probabilities</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class='profile-card'>
          <h3>Engineered Features</h3>
          <div class='subtitle'>12 metrics derived from raw stats</div>
          <div class='info-box'>
            <b>Rate metrics:</b> Podium Freq · Win Rate · Pole Freq · FL Freq · PPR<br><br>
            <b>Count metrics:</b> Wins · Podiums · Poles · Fastest Laps · Points · Races<br><br>
            <b>Consistency:</b> 1 / (1 + σ of finishing positions)
          </div>
        </div>""", unsafe_allow_html=True)

    # Limitations
    st.markdown("<div class='section-title'>Known Limitations</div>", unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("""
        <div class='profile-card'>
          <h3>Data Constraints</h3>
          <div class='subtitle'>What the model cannot see</div>
          <span class='pill-red'>Only 3 champions in training data</span>
          <span class='pill-red'>2025 based on 1–2 races</span>
          <span class='pill-red'>No injury / DNF cause data</span>
          <span class='pill-red'>No bike spec details</span>
          <span class='pill-red'>No weather / track data</span>
        </div>""", unsafe_allow_html=True)
    with lc2:
        st.markdown("""
        <div class='profile-card'>
          <h3>External Factors</h3>
          <div class='subtitle'>Events the model cannot predict</div>
          <span class='pill-red'>Injuries & accidents</span>
          <span class='pill-red'>Mid-season bike upgrades</span>
          <span class='pill-red'>Team strategy shifts</span>
          <span class='pill-red'>Regulatory changes</span>
          <span class='pill-red'>Intra-team dynamics</span>
        </div>""", unsafe_allow_html=True)

    # 2024 validation
    st.markdown("<div class='section-title'>2024 Season Validation</div>", unsafe_allow_html=True)
    val_df = pd.DataFrame({
        "Rider":            ["Jorge Martín", "Francesco Bagnaia", "Marc Márquez"],
        "Actual Finish":    ["🥇 1st",        "🥈 2nd",             "🥉 3rd"],
        "Model Prediction": ["81%",           "81%",               "2%"],
        "Correct?":         ["✅ Yes",         "✅ Yes",             "❌ No (satellite team)"],
    })
    st.dataframe(val_df, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='info-box'>
      The model correctly identified the top 2 contenders. Marc Márquez's low 2024 score
      reflected his satellite team status — his 2025 probability jumps to <b>34%</b> after
      moving to the factory Ducati squad.
    </div>""", unsafe_allow_html=True)


# ── PAGE: Future Forecast 2026-2030 ──────────────────────────────────────────
elif page == "📕 Future Forecast 2026–2030":

    YEARS_FUTURE  = [2025, 2026, 2027, 2028, 2029, 2030]
    DISPLACEMENTS = [1000, 950, 900, 850]
    CC_LABELS     = {1000: "1000cc (Current)", 950: "950cc", 900: "900cc", 850: "850cc"}
    CC_COLORS     = {1000: "#c8102e", 950: "#e07030", 900: "#d4a020", 850: "#4caf7d"}
    FLAG_MAP      = {
        "Spain": "🇪🇸", "Italy": "🇮🇹", "France": "🇫🇷", "Australia": "🇦🇺",
        "South Africa": "🇿🇦", "Japan": "🇯🇵", "Portugal": "🇵🇹", "Thailand": "🇹🇭",
    }

    st.markdown("""
    <div class='hero'>
      <h1>Future Championship Forecast</h1>
      <p>Projected winners for 2025–2030 across four engine displacement scenarios</p>
      <span class='badge'>2025–2030</span>
      <span class='badge'>1000cc → 850cc</span>
      <span class='badge'>Age-decay model</span>
      <span class='badge'>Displacement modifier</span>
    </div>""", unsafe_allow_html=True)

    if fdf_future.empty:
        st.error("Future predictions not found. Run `generate_future_predictions.py` first.")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Scenario Controls</div>", unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sel_cc = st.selectbox(
            "Engine Displacement",
            DISPLACEMENTS,
            format_func=lambda x: CC_LABELS[x],
            index=0,
        )
    with ctrl2:
        sel_year = st.selectbox("Season", YEARS_FUTURE, index=0)
    with ctrl3:
        top_n = st.slider("Show top N riders", 3, 10, 5)

    # ── Displacement explainer ────────────────────────────────────────────────
    cc_desc = {
        1000: "Current regulations. Raw power advantage favours factory Ducati and KTM.",
        950:  "Mild reduction. Slight compression of the field; technically precise riders gain ~5%.",
        900:  "Moderate reduction. Power advantage shrinks significantly; smooth riding style rewarded.",
        850:  "Maximum reduction modelled. Field tightest here; technical skill dominates over raw power.",
    }
    st.markdown(f"""
    <div class='info-box'>
      <b>{CC_LABELS[sel_cc]}</b> — {cc_desc[sel_cc]}
    </div>""", unsafe_allow_html=True)

    # ── Podium for selected year + cc ─────────────────────────────────────────
    st.markdown(f"<div class='section-title'>Predicted Podium — {sel_year} · {CC_LABELS[sel_cc]}</div>",
                unsafe_allow_html=True)

    slice_df = (fdf_future[(fdf_future["Year"] == sel_year) &
                            (fdf_future["Displacement_cc"] == sel_cc)]
                .sort_values("Championship_Probability", ascending=False)
                .reset_index(drop=True))

    top3_fut = slice_df.head(3)
    pod_styles = [("gold","🥇","1st"), ("silver","🥈","2nd"), ("bronze","🥉","3rd")]
    pcols = st.columns(3)
    for col, (style, medal, rank), (_, row) in zip(pcols, pod_styles, top3_fut.iterrows()):
        flag = FLAG_MAP.get(row["Country"], "")
        dm_delta = (row["Displacement_Modifier"] - 1) * 100
        dm_str   = f"+{dm_delta:.1f}%" if dm_delta >= 0 else f"{dm_delta:.1f}%"
        dm_color = "#4caf7d" if dm_delta >= 0 else "#e05070"
        with col:
            st.markdown(f"""
            <div class='podium-card {style}'>
              <div class='medal'>{medal}</div>
              <div style='font-size:0.75rem;font-weight:600;opacity:.7;'>{rank} PLACE</div>
              <div class='name'>{flag} {row['Rider']}</div>
              <div class='prob'>{row['Championship_Probability']:.1%}</div>
              {prob_bar(row['Championship_Probability'])}
              <div class='divider'></div>
              <div class='stat'>🎂 Age in {sel_year}: <b>{int(row['Age'])}</b></div>
              <div class='stat'>📈 Age factor: <b>{row['Age_Factor']:.2f}</b></div>
              <div class='stat' style='color:{dm_color};'>⚙️ Displacement Δ: <b>{dm_str}</b></div>
              <div class='stat'>🏁 Proj. Points: <b>{int(row['Proj_Points'])}</b></div>
              <div class='stat'>🏆 Proj. Wins: <b>{row['Proj_Wins']:.1f}</b></div>
              <div class='stat'>🎖️ Proj. Podiums: <b>{row['Proj_Podiums']:.1f}</b></div>
              <div class='team-tag'>{row['Motorcycle']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top-N bar chart ───────────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>Top {top_n} Riders — {sel_year} · {CC_LABELS[sel_cc]}</div>",
                unsafe_allow_html=True)

    top_n_df = slice_df.head(top_n).sort_values("Championship_Probability")
    fig, ax = plt.subplots(figsize=(10, max(3, top_n * 0.55)))
    bar_c = [CC_COLORS[sel_cc] if i == len(top_n_df) - 1 else
             "#8b3a4a" if i >= len(top_n_df) - 3 else "#2a1a20"
             for i in range(len(top_n_df))]
    bars = ax.barh(top_n_df["Rider"], top_n_df["Championship_Probability"] * 100,
                   color=bar_c, height=0.6)
    for bar, val in zip(bars, top_n_df["Championship_Probability"] * 100):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e8eaf0", fontsize=9, fontweight="bold")
    ax.set_xlabel("Championship Probability (%)", color="#7a7f96")
    ax.set_xlim(0, top_n_df["Championship_Probability"].max() * 130)
    ax.set_title(f"Championship Probability — {sel_year} · {CC_LABELS[sel_cc]}",
                 color="#e8eaf0", fontsize=12, fontweight="bold")
    dark_chart(fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Multi-year trend for top 5 riders (selected cc) ───────────────────────
    st.markdown(f"<div class='section-title'>Probability Trend 2025–2030 · {CC_LABELS[sel_cc]}</div>",
                unsafe_allow_html=True)

    # Identify top 5 riders by average probability across all years at this cc
    cc_all_years = fdf_future[fdf_future["Displacement_cc"] == sel_cc]
    top5_riders  = (cc_all_years.groupby("Rider")["Championship_Probability"]
                    .mean().nlargest(5).index.tolist())

    trend_colors = ["#c8102e","#4a90d9","#4caf7d","#f0a500","#9b59b6"]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for rider, color in zip(top5_riders, trend_colors):
        rdata = (cc_all_years[cc_all_years["Rider"] == rider]
                 .sort_values("Year"))
        ax2.plot(rdata["Year"], rdata["Championship_Probability"] * 100,
                 marker="o", linewidth=2.5, markersize=6, color=color, label=rider)
        # annotate last point
        last = rdata.iloc[-1]
        ax2.annotate(f"{last['Championship_Probability']:.0%}",
                     (last["Year"], last["Championship_Probability"] * 100),
                     textcoords="offset points", xytext=(6, 0),
                     color=color, fontsize=8, fontweight="bold")

    ax2.set_xlabel("Season", color="#7a7f96")
    ax2.set_ylabel("Championship Probability (%)", color="#7a7f96")
    ax2.set_title(f"Probability Trend — {CC_LABELS[sel_cc]}", color="#e8eaf0",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(YEARS_FUTURE)
    ax2.legend(fontsize=8, framealpha=0, labelcolor="#c8cad4",
               loc="upper left", bbox_to_anchor=(0, 1))
    dark_chart(fig2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Displacement comparison heatmap for selected year ─────────────────────
    st.markdown(f"<div class='section-title'>Displacement Comparison — {sel_year}</div>",
                unsafe_allow_html=True)

    # Build pivot: riders × displacements
    year_slice = fdf_future[fdf_future["Year"] == sel_year]
    pivot = (year_slice.pivot_table(index="Rider", columns="Displacement_cc",
                                    values="Championship_Probability")
             .fillna(0))
    # Keep only riders with any meaningful probability
    pivot = pivot[pivot.max(axis=1) > 0.01].sort_values(1000, ascending=False).head(12)
    pivot.columns = [CC_LABELS[c] for c in pivot.columns]

    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(pivot) * 0.55)))
    sns.heatmap(
        pivot * 100,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#0d0f14",
        cbar_kws={"label": "Probability (%)"},
        ax=ax3,
    )
    ax3.set_title(f"Championship Probability by Displacement — {sel_year}",
                  color="#e8eaf0", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Engine Displacement", color="#7a7f96")
    ax3.set_ylabel("")
    ax3.tick_params(colors="#c8cad4")
    fig3.patch.set_facecolor("#1a1d27")
    ax3.set_facecolor("#1a1d27")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Side-by-side cc comparison for selected year ──────────────────────────
    st.markdown(f"<div class='section-title'>Winner Probability Shift by Displacement — {sel_year}</div>",
                unsafe_allow_html=True)

    cc_cols = st.columns(4)
    for col, cc in zip(cc_cols, DISPLACEMENTS):
        sub = (fdf_future[(fdf_future["Year"] == sel_year) &
                           (fdf_future["Displacement_cc"] == cc)]
               .sort_values("Championship_Probability", ascending=False)
               .head(3))
        with col:
            st.markdown(f"""
            <div class='profile-card' style='padding:16px;'>
              <div style='color:{CC_COLORS[cc]};font-weight:800;font-size:1rem;
                          margin-bottom:10px;'>{CC_LABELS[cc]}</div>""",
                        unsafe_allow_html=True)
            for rank_i, (_, r) in enumerate(sub.iterrows(), 1):
                medals = ["🥇","🥈","🥉"]
                flag   = FLAG_MAP.get(r["Country"], "")
                st.markdown(f"""
              <div style='display:flex;justify-content:space-between;
                          align-items:center;padding:6px 0;
                          border-bottom:1px solid #252836;'>
                <span style='font-size:0.85rem;'>{medals[rank_i-1]} {flag} {r['Rider'].split()[-1]}</span>
                <span style='font-weight:800;color:{CC_COLORS[cc]};font-size:0.9rem;'>
                  {r['Championship_Probability']:.1%}</span>
              </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Age trajectory chart ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Rider Age Trajectory 2025–2030</div>",
                unsafe_allow_html=True)

    age_riders = top5_riders
    age_data   = fdf_future[(fdf_future["Displacement_cc"] == 1000) &
                              (fdf_future["Rider"].isin(age_riders))].copy()

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))

    for rider, color in zip(age_riders, trend_colors):
        rdata = age_data[age_data["Rider"] == rider].sort_values("Year")
        ax4a.plot(rdata["Year"], rdata["Age_Factor"], marker="o",
                  linewidth=2, markersize=5, color=color, label=rider)
        ax4b.plot(rdata["Year"], rdata["Age"], marker="s",
                  linewidth=2, markersize=5, color=color, label=rider)

    ax4a.set_title("Age Performance Factor", color="#e8eaf0", fontsize=11, fontweight="bold")
    ax4a.set_ylabel("Factor (1.0 = peak)", color="#7a7f96")
    ax4a.set_xlabel("Season", color="#7a7f96")
    ax4a.axhline(1.0, color="#7a7f96", linewidth=0.8, linestyle="--")
    ax4a.set_xticks(YEARS_FUTURE)
    ax4a.legend(fontsize=7, framealpha=0, labelcolor="#c8cad4")

    ax4b.set_title("Rider Age", color="#e8eaf0", fontsize=11, fontweight="bold")
    ax4b.set_ylabel("Age (years)", color="#7a7f96")
    ax4b.set_xlabel("Season", color="#7a7f96")
    ax4b.axhspan(26, 32, alpha=0.08, color="#4caf7d", label="Peak window (26–32)")
    ax4b.set_xticks(YEARS_FUTURE)
    ax4b.legend(fontsize=7, framealpha=0, labelcolor="#c8cad4")

    for ax in [ax4a, ax4b]:
        dark_chart(plt.figure())  # dummy; apply manually
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#7a7f96")
        for spine in ax.spines.values():
            spine.set_edgecolor("#252836")
    fig4.patch.set_facecolor("#1a1d27")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── Full data table ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Full Predictions Table</div>",
                unsafe_allow_html=True)

    tbl_cc   = st.selectbox("Filter by displacement", DISPLACEMENTS,
                             format_func=lambda x: CC_LABELS[x], key="tbl_cc")
    tbl_year = st.selectbox("Filter by year", YEARS_FUTURE, key="tbl_year")

    tbl_df = (fdf_future[(fdf_future["Year"] == tbl_year) &
                          (fdf_future["Displacement_cc"] == tbl_cc)]
              .sort_values("Championship_Probability", ascending=False)
              .reset_index(drop=True))
    tbl_df.insert(0, "Rank", tbl_df.index + 1)
    tbl_df["Prob %"]  = (tbl_df["Championship_Probability"] * 100).round(2)
    tbl_df["Disp. Δ"] = tbl_df["Displacement_Modifier"].apply(
        lambda x: f"+{(x-1)*100:.1f}%" if x >= 1 else f"{(x-1)*100:.1f}%")

    show_cols = ["Rank","Rider","Country","Motorcycle","Age","Age_Factor",
                 "Disp. Δ","Proj_Points","Proj_Wins","Proj_Podiums","Prob %"]
    st.dataframe(
        tbl_df[show_cols].style
            .background_gradient(subset=["Prob %"], cmap="Reds")
            .background_gradient(subset=["Proj_Points"], cmap="Blues")
            .format({"Age_Factor": "{:.2f}", "Proj_Points": "{:.0f}",
                     "Proj_Wins": "{:.1f}", "Proj_Podiums": "{:.1f}",
                     "Prob %": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # Download
    csv_fut = tbl_df[show_cols].to_csv(index=False)
    st.download_button(
        f"⬇️  Download {tbl_year} · {CC_LABELS[tbl_cc]} CSV",
        data=csv_fut,
        file_name=f"motogp_{tbl_year}_{tbl_cc}cc_predictions.csv",
        mime="text/csv",
    )

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>How Future Predictions Work</div>",
                unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""
        <div class='profile-card'>
          <h3>Age-Decay Model</h3>
          <div class='subtitle'>Performance vs rider age</div>
          <div class='info-box'>
            <b>Developing (< 24):</b> 88–96% of peak<br>
            <b>Peak window (26–32):</b> 100%<br>
            <b>Gentle decline (33–36):</b> −4% per year<br>
            <b>Steep decline (37+):</b> −7% per year, floor 60%
          </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class='profile-card'>
          <h3>Displacement Modifier</h3>
          <div class='subtitle'>How engine size shifts the field</div>
          <div class='info-box'>
            Smaller engines compress power gaps.<br>
            Riders with high <b>technical skill scores</b>
            (smooth, precise style) gain up to <b>+15%</b>
            at 850cc vs 1000cc.<br>
            Power-dependent riders lose up to <b>−5%</b>.
          </div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div class='profile-card'>
          <h3>Projection Assumptions</h3>
          <div class='subtitle'>What the model assumes</div>
          <div class='info-box'>
            • Team tier stays roughly constant<br>
            • Motorcycle brand unchanged from 2025<br>
            • No major injuries modelled<br>
            • Small random noise (±4%) per year<br>
            • Probabilities normalised to sum to 1
          </div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  🏍️ MotoGP 2025 Championship Predictor &nbsp;·&nbsp;
  Random Forest Classifier &nbsp;·&nbsp;
  Accuracy 81.82% &nbsp;·&nbsp; ROC-AUC 0.90 &nbsp;·&nbsp;
  Data: 2013–2024 seasons
</div>
""", unsafe_allow_html=True)
