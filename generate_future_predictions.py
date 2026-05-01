"""
generate_future_predictions.py
Generates championship probability predictions for 2026-2030 across
engine displacement scenarios: 1000cc, 950cc, 900cc, 850cc.

Logic:
  - Train a Random Forest on 2019-2024 MotoGP data (more seasons = better signal)
  - For each future year, project rider stats using a decay/growth model
    that accounts for age, recent form, team tier, and displacement impact
  - Displacement modifier: smaller engines compress the field (less top-end
    power advantage), benefiting technically precise riders
  - Output: data/future_predictions.csv  (year × displacement × rider)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings, json
warnings.filterwarnings("ignore")

# ── 1. Load & engineer features ───────────────────────────────────────────────
df = pd.read_csv("data/RidersSummary.csv")
motogp = df[df["class"] == "MotoGP"].copy()

def engineer(df):
    rows = []
    for (rider, season), g in df.groupby(["rider_name", "season"]):
        races   = g["races_participated"].sum()
        wins    = g["wins"].sum()
        podiums = g["podium"].sum()
        poles   = g["pole"].sum()
        fl      = g["fastest_lap"].sum()
        pts     = g["points"].sum()
        placed  = g["placed"].mean() if g["placed"].sum() > 0 else np.nan
        rows.append({
            "rider":          rider,
            "season":         season,
            "races":          races,
            "wins":           wins,
            "podiums":        podiums,
            "poles":          poles,
            "fastest_laps":   fl,
            "points":         pts,
            "avg_pos":        placed,
            "podium_freq":    podiums / races if races > 0 else 0,
            "win_rate":       wins    / races if races > 0 else 0,
            "pole_freq":      poles   / races if races > 0 else 0,
            "fl_freq":        fl      / races if races > 0 else 0,
            "consistency":    1 / (1 + g["placed"].std()) if g["placed"].std() > 0 else 1,
            "ppr":            pts     / races if races > 0 else 0,
            "is_champion":    1 if g["world_championships"].sum() > 0 else 0,
            "team":           g["team"].iloc[0],
            "motorcycle":     g["motorcycle"].iloc[0],
            "country":        g["home_country"].iloc[0],
        })
    return pd.DataFrame(rows)

feat = engineer(motogp)

FEATURE_COLS = ["races","wins","podiums","poles","fastest_laps","points",
                "podium_freq","win_rate","pole_freq","fl_freq","consistency","ppr"]

# ── 2. Train model on 2019-2024 ───────────────────────────────────────────────
train = feat[feat["season"].between(2019, 2024)].copy()
X = train[FEATURE_COLS].fillna(0)
y = train["is_champion"]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                             class_weight="balanced", random_state=42)
rf.fit(Xs, y)

# ── 3. Rider roster & metadata for 2025+ ─────────────────────────────────────
# Birth years for age-decay calculation
BIRTH_YEAR = {
    "Marc Marquez":          1993,
    "Francesco Bagnaia":     1997,
    "Jorge Martin":          1998,
    "Fabio Quartararo":      1999,
    "Enea Bastianini":       1997,
    "Brad Binder":           1995,
    "Alex Marquez":          1996,
    "Maverick Vinales":      1994,
    "Jack Miller":           1995,
    "Johann Zarco":          1990,
    "Franco Morbidelli":     1994,
    "Joan Mir":              1997,
    "Alex Rins":             1995,
    "Luca Marini":           1997,
    "Marco Bezzecchi":       1998,
    "Fabio Di Giannantonio": 1998,
    "Pedro Acosta":          2004,
    "Raul Fernandez":        2000,
    "Miguel Oliveira":       1994,
    "Ai Ogura":              2001,
    "Fermin Aldeguer":       2003,
    "Somkiat Chantra":       1999,
}

# Team tier: 1=factory, 2=top satellite, 3=satellite, 4=backmarker
TEAM_TIER_2025 = {
    "Marc Marquez":          1,
    "Francesco Bagnaia":     1,
    "Enea Bastianini":       3,   # moved to KTM Tech3
    "Pedro Acosta":          1,   # KTM factory
    "Brad Binder":           1,
    "Maverick Vinales":      3,
    "Jorge Martin":          2,   # Aprilia factory
    "Marco Bezzecchi":       2,   # Aprilia factory
    "Ai Ogura":              2,   # Trackhouse Aprilia
    "Fabio Quartararo":      2,   # Yamaha factory
    "Alex Rins":             2,
    "Jack Miller":           2,   # Pramac Yamaha
    "Miguel Oliveira":       2,
    "Alex Marquez":          2,   # Gresini Ducati
    "Fabio Di Giannantonio": 2,
    "Franco Morbidelli":     2,
    "Johann Zarco":          3,
    "Joan Mir":              3,
    "Luca Marini":           3,
    "Raul Fernandez":        2,
    "Fermin Aldeguer":       2,
    "Somkiat Chantra":       3,
}

# Motorcycle assignments 2025 (used as base; may change per year)
MOTO_2025 = {
    "Marc Marquez":          "Ducati",
    "Francesco Bagnaia":     "Ducati",
    "Enea Bastianini":       "KTM",
    "Pedro Acosta":          "KTM",
    "Brad Binder":           "KTM",
    "Maverick Vinales":      "KTM",
    "Jorge Martin":          "Aprilia",
    "Marco Bezzecchi":       "Aprilia",
    "Ai Ogura":              "Aprilia",
    "Fabio Quartararo":      "Yamaha",
    "Alex Rins":             "Yamaha",
    "Jack Miller":           "Yamaha",
    "Miguel Oliveira":       "Yamaha",
    "Alex Marquez":          "Ducati",
    "Fabio Di Giannantonio": "Ducati",
    "Franco Morbidelli":     "Ducati",
    "Johann Zarco":          "Honda",
    "Joan Mir":              "Honda",
    "Luca Marini":           "Honda",
    "Raul Fernandez":        "Aprilia",
    "Fermin Aldeguer":       "Ducati",
    "Somkiat Chantra":       "Honda",
}

# ── 4. Displacement modifier ──────────────────────────────────────────────────
# At 1000cc: current baseline (modifier = 1.0 for all)
# As cc drops, raw power advantage shrinks → technically precise riders gain
# We model this as a per-rider "technical skill" score that scales with cc reduction

TECHNICAL_SKILL = {
    # Higher = benefits more from smaller engines (smooth, precise riders)
    "Marc Marquez":          0.95,
    "Francesco Bagnaia":     0.90,
    "Fabio Quartararo":      0.92,
    "Pedro Acosta":          0.88,
    "Jorge Martin":          0.85,
    "Enea Bastianini":       0.80,
    "Alex Marquez":          0.82,
    "Brad Binder":           0.75,  # aggressive style, benefits less
    "Maverick Vinales":      0.83,
    "Marco Bezzecchi":       0.78,
    "Ai Ogura":              0.80,
    "Alex Rins":             0.79,
    "Jack Miller":           0.72,
    "Franco Morbidelli":     0.76,
    "Johann Zarco":          0.74,
    "Joan Mir":              0.77,
    "Luca Marini":           0.70,
    "Raul Fernandez":        0.73,
    "Miguel Oliveira":       0.76,
    "Fabio Di Giannantonio": 0.74,
    "Fermin Aldeguer":       0.82,
    "Somkiat Chantra":       0.75,
}

def displacement_modifier(rider, cc):
    """
    Returns a multiplier on the rider's base probability.
    At 1000cc → 1.0 (no change).
    At 850cc  → technical riders get up to +15%, power riders lose up to -10%.
    """
    if cc == 1000:
        return 1.0
    skill = TECHNICAL_SKILL.get(rider, 0.75)
    # Linear interpolation: 1000cc=0 effect, 850cc=full effect
    t = (1000 - cc) / 150          # 0 at 1000cc, 1.0 at 850cc
    # Skill 1.0 → +15% boost; skill 0.7 → -10% penalty
    effect = (skill - 0.75) * 1.0  # range roughly -0.05 to +0.20
    return 1.0 + t * effect

# ── 5. Age-decay model ────────────────────────────────────────────────────────
def age_factor(rider, year):
    """
    Peak performance window: 26-32.
    Slight growth before peak, gradual decay after.
    """
    age = year - BIRTH_YEAR.get(rider, 1995)
    if age < 24:
        return 0.88 + (age - 20) * 0.03   # still developing
    elif age <= 32:
        return 1.0                          # peak
    elif age <= 36:
        return 1.0 - (age - 32) * 0.04    # gentle decline
    else:
        return max(0.60, 1.0 - (age - 32) * 0.07)  # steeper decline

# ── 6. Base stats from 2024 season ───────────────────────────────────────────
base_2024 = feat[feat["season"] == 2024].set_index("rider").to_dict("index")

# For riders not in 2024 MotoGP, use their most recent season
for rider in BIRTH_YEAR:
    if rider not in base_2024:
        rider_hist = feat[feat["rider"] == rider]
        if not rider_hist.empty:
            base_2024[rider] = rider_hist.sort_values("season").iloc[-1].to_dict()
        else:
            # Rookie defaults
            base_2024[rider] = {c: 0 for c in FEATURE_COLS}
            base_2024[rider].update({"races": 20, "ppr": 5, "consistency": 0.8})

# ── 7. Project stats for future years ────────────────────────────────────────
def project_stats(rider, year, cc):
    """
    Project a rider's feature vector for a given year and displacement.
    """
    base = base_2024.get(rider, {c: 0 for c in FEATURE_COLS})
    af   = age_factor(rider, year)
    dm   = displacement_modifier(rider, cc)
    tier = TEAM_TIER_2025.get(rider, 3)
    tier_boost = {1: 1.10, 2: 1.00, 3: 0.90, 4: 0.75}.get(tier, 1.0)

    # Year-over-year form trend (small random noise for realism)
    rng  = np.random.default_rng(seed=hash((rider, year, cc)) % (2**31))
    noise = rng.normal(1.0, 0.04)

    scale = af * dm * tier_boost * noise

    projected = {}
    for col in FEATURE_COLS:
        val = float(base.get(col, 0) or 0)
        projected[col] = max(0, val * scale)

    # Clamp rates to [0, 1]
    for rate_col in ["podium_freq","win_rate","pole_freq","fl_freq","consistency"]:
        projected[rate_col] = min(1.0, projected[rate_col])

    return projected

# ── 8. Generate predictions ───────────────────────────────────────────────────
YEARS = [2025, 2026, 2027, 2028, 2029, 2030]
DISPLACEMENTS = [1000, 950, 900, 850]

riders = list(BIRTH_YEAR.keys())
records = []

for year in YEARS:
    for cc in DISPLACEMENTS:
        rows_feat = []
        for rider in riders:
            proj = project_stats(rider, year, cc)
            rows_feat.append(proj)

        X_proj = pd.DataFrame(rows_feat, columns=FEATURE_COLS).fillna(0)
        X_proj_s = scaler.transform(X_proj)
        probs = rf.predict_proba(X_proj_s)[:, 1]

        # Normalise so probabilities sum to 1 (one champion per season)
        total = probs.sum()
        if total > 0:
            probs_norm = probs / total
        else:
            probs_norm = np.ones(len(probs)) / len(probs)

        for rider, raw_p, norm_p, feat_row in zip(riders, probs, probs_norm, rows_feat):
            records.append({
                "Year":                    year,
                "Displacement_cc":         cc,
                "Rider":                   rider,
                "Country":                 feat[feat["rider"] == rider]["country"].iloc[-1]
                                           if not feat[feat["rider"] == rider].empty else "Unknown",
                "Motorcycle":              MOTO_2025.get(rider, "Unknown"),
                "Team_Tier":               TEAM_TIER_2025.get(rider, 3),
                "Age":                     year - BIRTH_YEAR.get(rider, 1995),
                "Age_Factor":              round(age_factor(rider, year), 3),
                "Displacement_Modifier":   round(displacement_modifier(rider, cc), 3),
                "Raw_Probability":         round(float(raw_p), 6),
                "Championship_Probability":round(float(norm_p), 6),
                # Projected stats (rounded for readability)
                "Proj_Points":             round(feat_row["points"]),
                "Proj_Wins":               round(feat_row["wins"], 1),
                "Proj_Podiums":            round(feat_row["podiums"], 1),
                "Proj_Poles":              round(feat_row["poles"], 1),
                "Proj_PPR":                round(feat_row["ppr"], 2),
                "Proj_Podium_Freq":        round(feat_row["podium_freq"], 3),
                "Proj_Win_Rate":           round(feat_row["win_rate"], 3),
                "Proj_Consistency":        round(feat_row["consistency"], 3),
            })

out = pd.DataFrame(records)
out.to_csv("data/future_predictions.csv", index=False)
print(f"✓ Saved data/future_predictions.csv  ({len(out)} rows)")

# ── 9. Quick sanity check ─────────────────────────────────────────────────────
print("\nTop 3 per year/displacement (1000cc):")
for year in YEARS:
    sub = out[(out["Year"] == year) & (out["Displacement_cc"] == 1000)]
    top3 = sub.nlargest(3, "Championship_Probability")[["Rider","Championship_Probability","Age"]]
    print(f"\n  {year}:")
    for _, r in top3.iterrows():
        print(f"    {r['Rider']:25s}  {r['Championship_Probability']:.1%}  (age {int(r['Age'])})")

print("\nTop 3 per year/displacement (850cc):")
for year in YEARS:
    sub = out[(out["Year"] == year) & (out["Displacement_cc"] == 850)]
    top3 = sub.nlargest(3, "Championship_Probability")[["Rider","Championship_Probability","Age"]]
    print(f"\n  {year}:")
    for _, r in top3.iterrows():
        print(f"    {r['Rider']:25s}  {r['Championship_Probability']:.1%}  (age {int(r['Age'])})")
