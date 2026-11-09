import argparse
import numpy as np
import pandas as pd

SYMPTOMS = ["anxiety", "insomnia", "dizziness", "irritability",
            "nausea", "headache", "low_mood", "fatigue"]
# How strongly each symptom tracks the hidden "distress" level (assumption)
LOAD = dict(anxiety=1.0, insomnia=0.9, dizziness=0.7, irritability=0.6,
            nausea=0.5, headache=0.5, low_mood=0.7, fatigue=0.6)
# Typical baseline severity before the taper starts (assumption)
BASE = dict(anxiety=0.5, insomnia=0.6, dizziness=0.1, irritability=0.4,
            nausea=0.1, headache=0.3, low_mood=0.5, fatigue=0.5)
# Relative withdrawal tendency by drug (assumption, loosely: paroxetine and
# venlafaxine rougher, fluoxetine gentler)
DRUGS = {  # name: (class, factor)
    "paroxetine": ("SSRI", 1.4), "sertraline": ("SSRI", 1.0),
    "escitalopram": ("SSRI", 0.9), "fluoxetine": ("SSRI", 0.6),
    "venlafaxine": ("SNRI", 1.3), "duloxetine": ("SNRI", 1.1),
}
# Dose steps as % of the starting dose (relative dose, not mg)
SCHEDULES = {
    "standard": [100, 75, 50, 25, 0],
    "gradual": [100, 80, 60, 45, 30, 20, 10, 5, 0],
}
SEV_WORD = {1: "a touch of", 2: "some", 3: "quite a bit of", 4: "severe"}
PHRASE = {
    "anxiety": "anxiety", "insomnia": "trouble sleeping", "dizziness": "dizziness",
    "irritability": "irritability", "nausea": "nausea", "headache": "headaches",
    "low_mood": "low mood", "fatigue": "tiredness",
}


def feedback_text(sev, red_flag, rng):
    parts = [f"{SEV_WORD[v]} {PHRASE[s]}" for s, v in sev.items() if v >= 1]
    rng.shuffle(parts)
    if not parts:
        text = "Felt mostly fine this week."
    elif max(sev.values()) <= 1:
        text = "Mostly okay this week, with " + ", ".join(parts) + "."
    else:
        text = "This week I had " + ", ".join(parts[:4]) + "."
    if red_flag:
        text += " Had some dark thoughts and wanted to talk to someone."
    return text


def simulate_user(uid, rng, scale, theta, label_noise):
    drug = rng.choice(list(DRUGS))
    dclass, dfac = DRUGS[drug]
    sched_name = rng.choice(list(SCHEDULES), p=[0.55, 0.45])
    steps = SCHEDULES[sched_name]
    cadence = int(rng.choice([2, 3], p=[0.7, 0.3]))
    sens = rng.lognormal(mean=-0.1, sigma=0.5)          # personal sensitivity
    delay = int(rng.integers(0, 3))                      # weeks before effect shows
    tau = rng.uniform(1.5, 4.0)                          # how fast effect fades (weeks)
    base_mult = {s: rng.uniform(0.5, 1.5) for s in SYMPTOMS}

    n_steps = len(steps)
    horizon = (n_steps - 1) * cadence + 5
    dose = [steps[min((t - 1) // cadence, n_steps - 1)] if t > 0 else 100
            for t in range(horizon + 1)]
    # relative reduction at each change (cutting a lower dose by the same mg = bigger relative cut)
    rel_cut = {}
    for t in range(1, horizon + 1):
        if dose[t] < dose[t - 1]:
            rel_cut[t] = (dose[t - 1] - dose[t]) / dose[t - 1]

    rows, ar = [], 0.0
    last_change = 0
    for t in range(1, horizon + 1):
        if t in rel_cut:
            last_change = t
        effect = sum(r * np.exp(-(t - k - delay) / tau)
                     for k, r in rel_cut.items() if t - k - delay >= 0)
        ar = 0.5 * ar + rng.normal(0, 0.3)
        distress = scale * sens * dfac * effect + ar
        sev = {}
        for s in SYMPTOMS:
            x = BASE[s] * base_mult[s] + LOAD[s] * max(distress, 0) + rng.normal(0, 0.5)
            if rng.random() < 0.04:                      # unrelated random symptom flare
                x += rng.integers(1, 3)
            sev[s] = int(np.clip(round(x), 0, 4))
        p_red = 0.003 + 0.03 * max(distress - 2.0, 0)
        red = int(rng.random() < p_red)
        y = int(distress >= theta)
        if rng.random() < label_noise:
            y = 1 - y
        rows.append(dict(
            user_id=uid, drug=drug, drug_class=dclass, schedule=sched_name,
            week=t, dose_pct=dose[t],
            dose_change_pct=dose[t] - dose[t - 1],
            weeks_since_change=t - last_change if last_change else t,
            **{f"sym_{s}": sev[s] for s in SYMPTOMS},
            red_flag=red, feedback_text=feedback_text(sev, red, rng),
            label_negative=y,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.6, help="overall withdrawal strength")
    ap.add_argument("--theta", type=float, default=1.2, help="hidden distress level that makes a week negative")
    ap.add_argument("--label_noise", type=float, default=0.05)
    ap.add_argument("--out", default="synthetic_checkins.csv")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = []
    for uid in range(1, a.users + 1):
        rows += simulate_user(uid, rng, a.scale, a.theta, a.label_noise)
    df = pd.DataFrame(rows)
    users = df["user_id"].unique()
    test_users = set(rng.choice(users, size=int(0.2 * len(users)), replace=False))
    df["split"] = np.where(df["user_id"].isin(test_users), "test", "train")  # split by person
    df.to_csv(a.out, index=False)

    print(f"rows: {len(df)}  users: {df.user_id.nunique()}  weeks/user: {df.groupby('user_id').size().mean():.1f}")
    print(f"negative-week rate: {df.label_negative.mean():.1%}   red-flag rate: {df.red_flag.mean():.1%}")
    print("negative-week rate by drug:\n" + df.groupby('drug').label_negative.mean().round(3).to_string())
    sym = [f"sym_{s}" for s in SYMPTOMS]
    tot = df[sym].sum(axis=1)
    print("mean total symptom severity (pos vs neg weeks):",
          tot.groupby(df.label_negative).mean().round(2).to_dict())
    print("share of positive weeks with at least one symptom >=2:",
          round(((df[df.label_negative == 0][sym] >= 2).any(axis=1)).mean(), 3))


if __name__ == "__main__":
    main()
