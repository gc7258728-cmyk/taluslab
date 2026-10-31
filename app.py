# -*- coding: utf-8 -*-
"""TalusLab : atelier pédagogique de pente infinie. Tout est contenu dans ce fichier.

Les données restent dans la session du navigateur ; aucun fichier n'est écrit par
l'application. Exporter le carnet JSON pour le conserver. Validé sous Python 3.14.
"""
from __future__ import annotations

import copy
import hashlib
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

VERSION = "1.0"
PETROL, COPPER, SAND, RED = "#153F49", "#A24F2A", "#E9D2AD", "#AE3A36"
SOURCES = [
    ("GeoEq — modèle de pente infinie (équations et exemple sec)",
     "https://docs.geoeq.org/modules/design/slopes/"),
    ("University of the West of England — mécanique des pentes, pression interstitielle",
     "https://environment.uwe.ac.uk/geocal/slopes/slopmech.htm"),
]
BASE = dict(beta=20.0, phi=30.0, c=0.0, gamma=20.0, z=2.0, u=0.0)
RANGES = dict(beta=(1.0, 75.0), phi=(0.0, 60.0), c=(0.0, 100.0),
              gamma=(10.0, 30.0), z=(0.2, 20.0), u=(0.0, 300.0))
LABELS = {"beta": "Angle de pente β (°)", "phi": "Frottement effectif φ′ (°)",
          "c": "Cohésion effective c′ (kPa)", "gamma": "Poids volumique γ (kN/m³)",
          "z": "Profondeur verticale z (m)", "u": "Pression interstitielle u (kPa)"}
SOIL_COLS = ["id", "serie", "phi_deg", "c_kpa", "gamma_kn_m3", "origine"]
SOIL_NUM = {"phi_deg": (0, 60), "c_kpa": (0, 100), "gamma_kn_m3": (10, 30)}
SOIL_NAMES = {"phi_deg": "Frottement φ′ (°)", "c_kpa": "Cohésion c′ (kPa)",
              "gamma_kn_m3": "Poids volumique γ (kN/m³)"}
LIMITS = ("Pente infinie homogène ; plan parallèle à la surface ; z est une profondeur "
          "verticale. Calcul en contraintes effectives, statique, sans surcharge, séisme, "
          "succion ni rupture circulaire. u est une pression au plan étudié, pas une pluie. "
          "F > 1 ne constitue pas une validation de sécurité ni une conformité réglementaire.")


def fmt(value, digits=2):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def validated_params(params):
    """Les bornes sont des limites de saisie pédagogiques, pas des normes."""
    if not isinstance(params, dict) or set(params) != set(BASE):
        raise ValueError("Il faut exactement β, φ′, c′, γ, z et u.")
    result = {}
    for key, (lo, hi) in RANGES.items():
        if isinstance(params[key], bool):
            raise ValueError(f"{LABELS[key]} : nombre attendu.")
        try:
            val = float(params[key])
        except (TypeError, ValueError):
            raise ValueError(f"{LABELS[key]} : nombre attendu.") from None
        if not math.isfinite(val) or not lo <= val <= hi:
            raise ValueError(f"{LABELS[key]} : saisir une valeur entre {lo} et {hi}.")
        result[key] = val
    normal = result["gamma"] * result["z"] * math.cos(math.radians(result["beta"])) ** 2
    if result["u"] > normal + 1e-10:
        raise ValueError(f"u dépasse la contrainte normale ({fmt(normal)} kPa) : "
                         "ce modèle ne traite pas la perte de contact. Réduire u ou revoir les hypothèses.")
    return result


def calculate(params):
    p = validated_params(params)
    b, phi = math.radians(p["beta"]), math.radians(p["phi"])
    normal = p["gamma"] * p["z"] * math.cos(b) ** 2
    shear = p["gamma"] * p["z"] * math.sin(b) * math.cos(b)
    effective = max(0.0, normal - p["u"])
    strength = p["c"] + effective * math.tan(phi)
    return dict(normal=normal, effective=effective, shear=shear,
                strength=strength, F=strength / shear)


def interpretation(factor):
    if abs(factor - 1) < 1e-9:
        return "Équilibre limite du modèle", "amber"
    if factor < 1:
        return "Résistance insuffisante dans le modèle", "red"
    return "Résistance supérieure aux efforts calculés", "green"


def theoretical_probability(beta, lo, hi):
    """Uniquement : φ uniforme, c=u=0, β fixe ; événement strict F<1."""
    if lo > hi:
        raise ValueError("Bornes inversées.")
    if lo == hi:
        return float(lo < beta)
    return float(np.clip((beta - lo) / (hi - lo), 0, 1))


def validate_sim(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration de simulation invalide.")
    required = {"base", "bounds", "law", "n", "seed"}
    if set(config) != required:
        raise ValueError("Configuration de simulation incomplète.")
    p = validated_params(config["base"])
    if config["law"] not in ("Uniforme", "Triangulaire"):
        raise ValueError("Loi inconnue.")
    if type(config["n"]) is not int or not 1000 <= config["n"] <= 100000:
        raise ValueError("Choisir entre 1 000 et 100 000 tirages.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2147483647:
        raise ValueError("Graine entière attendue entre 0 et 2 147 483 647.")
    if set(config["bounds"]) != {"phi", "c", "u"}:
        raise ValueError("Bornes de φ′, c′ et u attendues.")
    bounds = {}
    for key in ("phi", "c", "u"):
        v = config["bounds"][key]
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("Deux bornes attendues par variable.")
        a, b = map(float, v)
        low, high = RANGES[key]
        if not (math.isfinite(a) and math.isfinite(b) and low <= a <= b <= high):
            raise ValueError(f"Bornes impossibles pour {LABELS[key]}.")
        bounds[key] = [a, b]
    # Tous les tirages sont admissibles : aucun échantillon n'est supprimé.
    if bounds["u"][1] > calculate(p)["normal"] + 1e-10:
        raise ValueError("La borne haute de u dépasse la contrainte normale. "
                         "Réduire cette borne : aucun tirage ne sera éliminé silencieusement.")
    return dict(base=p, bounds=bounds, law=config["law"], n=config["n"], seed=config["seed"])


@st.cache_data(max_entries=16, show_spinner=False)
def simulate(config):
    cfg = validate_sim(config)
    rng = np.random.default_rng(cfg["seed"])
    values = {}
    for key, (a, b) in cfg["bounds"].items():
        if a == b:
            values[key] = np.full(cfg["n"], a)
        elif cfg["law"] == "Uniforme":
            values[key] = rng.uniform(a, b, cfg["n"])
        else:
            values[key] = rng.triangular(a, (a + b) / 2, b, cfg["n"])
    r = calculate(cfg["base"])
    factors = (values["c"] + (r["normal"] - values["u"]) *
               np.tan(np.radians(values["phi"]))) / r["shear"]
    trials = pd.DataFrame({"phi_deg": values["phi"], "c_kpa": values["c"],
                           "u_kpa": values["u"], "F": factors})
    k, n = int((factors < 1 - 1e-12).sum()), cfg["n"]
    prob = k / n
    z95 = stats.norm.ppf(0.975)
    denom = 1 + z95 ** 2 / n
    center = (prob + z95 ** 2 / (2 * n)) / denom
    radius = z95 * math.sqrt(prob * (1 - prob) / n + z95 ** 2 / (4 * n * n)) / denom
    exact = None
    if cfg["law"] == "Uniforme" and cfg["bounds"]["c"] == [0., 0.] and cfg["bounds"]["u"] == [0., 0.]:
        exact = theoretical_probability(cfg["base"]["beta"], *cfg["bounds"]["phi"])
    sensitivity = []
    for key in ("phi_deg", "c_kpa", "u_kpa"):
        if trials[key].nunique() > 1 and trials["F"].nunique() > 1:
            sensitivity.append(dict(variable=key, rho=float(stats.spearmanr(trials[key], trials["F"]).statistic)))
    return dict(config=cfg, probability=prob, count=k, n=n, ci=[max(0., center-radius), min(1., center+radius)],
                quantiles=[float(x) for x in np.quantile(factors, [.05, .5, .95])],
                mean=float(np.mean(factors)), exact=exact, sensitivity=sensitivity, trials=trials)


def example_soils():
    return [dict(id=f"E{i+1:02}", serie="Sable A — démonstration",
                 phi_deg=phi, c_kpa=c, gamma_kn_m3=g, origine="Fictif")
            for i, (phi, c, g) in enumerate(zip([27, 28, 29, 30, 31, 32, 33],
                                              [1, 2, 3, 4, 5, 6, 7],
                                              [19, 19.2, 19.4, 19.5, 19.6, 19.8, 20]))]


def clean_soils(records):
    raw = pd.DataFrame(records)
    if len(raw) > 20000:
        raise ValueError("Le carnet est limité à 20 000 lignes.")
    if raw.empty:
        return pd.DataFrame(columns=SOIL_COLS), pd.DataFrame(columns=["ligne", "motif"])
    missing = set(SOIL_NUM) - set(raw.columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    raw = raw.copy()
    for key, default in (("id", None), ("serie", "Série importée"), ("origine", "Déclaré par l’utilisateur")):
        if key not in raw:
            raw[key] = [f"I{i+1:05}" for i in range(len(raw))] if key == "id" else default
    issues, good, seen = [], [], set()
    for i, row in raw.iterrows():
        reasons, item = [], {}
        for col in ("id", "serie", "origine"):
            val = row[col]
            val = "" if pd.isna(val) else str(val).strip()
            if not val:
                reasons.append(col + " manquant")
            if len(val) > 160:
                reasons.append(col + " trop long (160 caractères maximum)")
            item[col] = val
        if item["id"] in seen:
            reasons.append("identifiant déjà présent")
        for col, (lo, hi) in SOIL_NUM.items():
            try:
                v = float(str(row[col]).strip().replace(",", "."))
                if not math.isfinite(v) or not lo <= v <= hi:
                    raise ValueError()
                item[col] = v
            except (ValueError, TypeError):
                reasons.append(f"{col} : nombre entre {lo} et {hi} attendu")
        if reasons:
            issues.append({"ligne": int(i)+1, "motif": " ; ".join(reasons)})
        else:
            seen.add(item["id"])
            good.append(item)
    return pd.DataFrame(good, columns=SOIL_COLS), pd.DataFrame(issues, columns=["ligne", "motif"])


def soil_summary(frame):
    if frame.empty:
        return pd.DataFrame(columns=["Variable", "n", "Moyenne", "Médiane", "Variance", "Écart-type", "CV (%)"])
    rows = []
    for col, label in SOIL_NAMES.items():
        s = frame[col].astype(float)
        mean, std = s.mean(), s.std(ddof=1)
        rows.append({"Variable": label, "n": len(s), "Moyenne": mean, "Médiane": s.median(),
                     "Variance": s.var(ddof=1), "Écart-type": std,
                     "CV (%)": 100*std/mean if mean != 0 else np.nan})
    return pd.DataFrame(rows)


def read_soil_file(blob, name):
    if len(blob) > 2_000_000:
        raise ValueError("Fichier trop volumineux (2 Mo maximum).")
    try:
        if name.lower().endswith(".xlsx"):
            frame = pd.read_excel(io.BytesIO(blob), engine="openpyxl", dtype=str)
        else:
            text = blob.decode("utf-8-sig")
            frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, keep_default_na=False)
        if len(frame) > 20000:
            raise ValueError("Maximum : 20 000 lignes.")
        records = frame.fillna("").to_dict("records")
        clean_soils(records)
        return records
    except (UnicodeDecodeError, KeyError, pd.errors.ParserError, ValueError, OSError, ImportError) as exc:
        raise ValueError("Import impossible : " + str(exc)) from None


def initial_sim():
    return dict(base={**BASE, "beta": 30.0}, bounds={"phi": [25., 35.], "c": [0., 0.], "u": [0., 0.]},
                law="Uniforme", n=10000, seed=42)


def validate_backup(data):
    if not isinstance(data, dict) or data.get("application") != "TalusLab" or data.get("version") != VERSION:
        raise ValueError("Sauvegarde TalusLab 1.0 attendue.")
    if data.get("mode") not in ("Découverte", "Étude détaillée"):
        raise ValueError("Mode inconnu.")
    p = validated_params(data.get("params"))
    if data["mode"] == "Découverte" and (p["c"] != 0 or p["u"] != 0):
        raise ValueError("Le mode découverte exige c′ = u = 0.")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) > 20:
        raise ValueError("Maximum : 20 scénarios.")
    names = set()
    for s in scenarios:
        if not isinstance(s, dict) or set(s) != {"nom", "params", "origine"}:
            raise ValueError("Scénario invalide.")
        if not isinstance(s["nom"], str) or not s["nom"].strip() or len(s["nom"]) > 80 or s["nom"] in names:
            raise ValueError("Nom de scénario vide, trop long ou répété.")
        if not isinstance(s["origine"], str) or len(s["origine"]) > 1000:
            raise ValueError("Origine du scénario invalide.")
        names.add(s["nom"])
        validated_params(s["params"])
    soils = data.get("soils")
    if not isinstance(soils, list) or any(not isinstance(r, dict) for r in soils):
        raise ValueError("Carnet de sol invalide.")
    clean_soils(soils)
    sim = None if data.get("simulation") is None else validate_sim(data["simulation"])
    origin = data.get("origin", "Origine non renseignée dans la sauvegarde")
    if not isinstance(origin, str) or len(origin) > 1000:
        raise ValueError("Origine du calcul invalide.")
    return dict(params=p, mode=data["mode"], origin=origin, scenarios=copy.deepcopy(scenarios),
                soils=copy.deepcopy(soils), simulation=sim)


def slope_geometry(params):
    """Projection à échelles x/y égales, profondeur z verticale, plan parallèle."""
    p = validated_params(params)
    length, height = 8., 8.*math.tan(math.radians(p["beta"]))
    scale = min(510 / length, 235 / (height+p["z"]+1))
    left = 100 + (510-length*scale)/2
    top = 95
    def xy(x, y):
        return (left+x*scale, top-y*scale)
    surface = [xy(0, 0), xy(length, -height)]
    plane = [xy(0, -p["z"]), xy(length, -height-p["z"])]
    return dict(surface=surface, plane=plane, scale=scale, height=height, length=length,
                polygon=[*surface, plane[1], plane[0]], beta=p["beta"], z=p["z"])


def slope_svg(params):
    p, g = validated_params(params), slope_geometry(params)
    pts = lambda seq: " ".join(f"{x:.2f},{y:.2f}" for x, y in seq)
    (ax, ay), (bx, by) = g["surface"]
    (_, cy), (_, dy) = g["plane"]
    midx = (ax+bx)/2
    midy = (cy+dy)/2
    nx, ny = -math.sin(math.radians(p["beta"])), -math.cos(math.radians(p["beta"]))
    water = ""
    if p["u"] > 0:
        for t in (.35, .55, .75):
            x, y = ax+(bx-ax)*t, cy+(dy-cy)*t
            water += f'<path d="M{x},{y} l{nx*28},{ny*28}" stroke="#28869D" stroke-width="4" marker-end="url(#water)"/>'
    return f'''<div class="terrain" role="img" aria-label="Coupe d'une pente infinie, plan parallèle, profondeur verticale {p['z']} mètres, pente {p['beta']} degrés.">
    <div class="terrain-top"><span>COUPE DU TERRAIN</span><span>01 / MODÈLE MÉCANIQUE</span></div>
    <svg viewBox="0 0 700 405" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs><pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="#d9e0d8" stroke-width=".7"/></pattern>
    <pattern id="soil" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="3" cy="4" r="1" fill="#ac8056" opacity=".35"/></pattern>
    <marker id="water" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0 0L5 3L0 6" fill="none" stroke="#28869D"/></marker></defs>
    <rect width="700" height="405" fill="url(#grid)"/>
    <text x="28" y="43" font-size="22" fill="{PETROL}" font-family="sans-serif">β = {fmt(p['beta'],1)}°</text>
    <text x="672" y="43" text-anchor="end" font-size="22" fill="{PETROL}" font-family="sans-serif">φ′ = {fmt(p['phi'],1)}°</text>
    <polygon points="{pts(g['polygon'])}" fill="{SAND}" stroke="none"/>
    <polygon points="{pts(g['polygon'])}" fill="url(#soil)"/>
    <line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{PETROL}" stroke-width="5"/>
    <line x1="{ax}" y1="{cy}" x2="{bx}" y2="{dy}" stroke="{COPPER}" stroke-width="3.5" stroke-dasharray="10 6"/>
    <path d="M{ax-22},{ay}H{ax-7} M{ax-15},{ay}V{cy} M{ax-22},{cy}H{ax-7}" stroke="{PETROL}" fill="none" stroke-width="2"/>
    <text x="{ax-30}" y="{(ay+cy)/2+6}" text-anchor="end" font-size="22" fill="{PETROL}" font-family="sans-serif">z</text>
    <path d="M{ax},{ay}h70" stroke="#64808a" stroke-dasharray="4 4"/>
    {water}
    <text x="28" y="375" font-size="19" fill="{PETROL}" font-family="sans-serif">Échelles x/y égales · fenêtre de pente longue</text>
    </svg><div class="terrain-legend"><span><i class="line"></i>Surface du sol</span><span><i class="line dashed"></i>Plan étudié</span><span>z : verticale</span></div></div>'''


CSS = """<style>
:root {--petrol:#153f49;--copper:#a24f2a;--cream:#f7f5ee;--ink:#163a43;--muted:#526568}
.stApp {background:#f7f5ee;color:var(--ink);font-family:'Segoe UI',sans-serif}
[data-testid="stHeader"] {background:rgba(247,245,238,.94)}
[data-testid="stMainBlockContainer"] {padding-top:4.7rem;max-width:1580px;padding-bottom:4rem}
h1,h2,h3 {color:var(--ink)!important;letter-spacing:-.035em}
h1 {font-size:clamp(2rem,3.7vw,3.3rem)!important;line-height:1.08!important}
[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p {color:var(--ink)}
[data-testid="stCaptionContainer"] {opacity:1!important;color:var(--muted)!important}
[data-testid="stCaptionContainer"] p {color:var(--muted)!important;opacity:1!important}
[data-testid="stSidebar"] {background:#153f49;border-right:1px solid #31535b}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebarNav"] a p,[data-testid="stSidebarNav"] a span,[data-testid="stSidebar"] svg {color:#f9f3e8!important}
[data-testid="stSidebarNav"] a {border-radius:9px;margin-bottom:6px;padding:10px}
[data-testid="stSidebarNav"] a[aria-current="page"] {background:#2d5d65;border-left:3px solid #e1aa7c}
.brand {font-size:2.1rem;font-weight:800;letter-spacing:-1px;color:#fff2df;line-height:1.2}
.brand b {color:#e9af7b}.side-note {color:#e0e8e3;font-size:.83rem;line-height:1.7;margin:18px 0}
.eyebrow {font-size:.74rem;font-weight:750;letter-spacing:.17em;color:var(--copper);text-transform:uppercase;margin-bottom:12px}
.intro {color:var(--muted);max-width:800px;font-size:1.04rem;margin-bottom:20px}
.terrain {background:#edf1e9;border:1px solid #c8d4ca;border-radius:20px;overflow:hidden;box-shadow:0 12px 35px #153f4910}
.terrain-top {display:flex;justify-content:space-between;gap:12px;padding:18px 22px;font-size:.69rem;letter-spacing:.12em;font-weight:750;border-bottom:1px solid #d3dcd2}
.terrain svg {width:100%;display:block;height:auto}
.st-key-terrain {background:#edf1e9;border:1px solid #c8d4ca!important;border-radius:20px!important;padding:0!important;overflow:hidden;box-shadow:0 12px 35px #153f4910}
.st-key-terrain [data-testid="stVerticalBlock"] {gap:0}
.st-key-terrain [data-testid="stImage"] {margin:0}
.terrain-legend {display:flex;gap:14px;flex-wrap:wrap;padding:14px 20px;background:#e4eadf;font-size:.78rem;color:#243f40}
.terrain-legend span {display:flex;align-items:center;gap:7px}.line {display:inline-block;width:22px;border-top:3px solid #153f49}.line.dashed {border-color:#a24f2a;border-top-style:dashed}
.result {background:#153f49;color:#fff5e7;border-radius:17px;padding:24px 26px;margin:16px 0 10px;display:flex;align-items:center;gap:26px;flex-wrap:wrap}
.result .number {font-size:3.1rem;font-weight:750;line-height:1;letter-spacing:-.06em}.result .label {font-size:.7rem;letter-spacing:.12em;color:#e1c8a5}.result .message {font-size:.95rem;max-width:340px;line-height:1.5;color:#fff5e7}
.result.red {background:#842f2c}.result.amber {background:#74501b}
.mini-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}
.mini {border-top:3px solid #b26e45;background:#fffcf5;padding:16px;border-radius:0 0 10px 10px;min-width:0}
.mini strong {font-size:1.5rem;letter-spacing:-.04em;display:block;color:#153f49}.mini small {color:#526568;font-size:.76rem}
.note {border-left:3px solid #b26e45;background:#eee8db;padding:14px 18px;color:#294149;font-size:.89rem;line-height:1.6;margin:16px 0}
button[kind="primary"],button[kind="primaryFormSubmit"] {background:#a24f2a!important;color:white!important;border:1px solid #a24f2a!important;border-radius:9px!important}
button[kind="primary"] p,button[kind="primaryFormSubmit"] p {color:white!important}
button[kind="secondary"],button[kind="secondaryFormSubmit"] {background:#fffdf7!important;border:1px solid #a8b9b4!important;border-radius:9px!important;color:#153f49!important}
button[kind="secondary"] p,button[kind="secondaryFormSubmit"] p {color:#153f49!important}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button {height:auto;min-height:2.6rem}
[data-testid="stButton"] p,[data-testid="stFormSubmitButton"] p,[data-testid="stDownloadButton"] p {white-space:normal!important;overflow:visible!important}
[data-testid="stSlider"] [role="slider"] {background:#a24f2a!important}
[data-testid="stSliderThumbValue"] {color:#153f49!important}
button[data-variant="segmented_control"][aria-checked="true"] {background:#153f49!important;color:#fff!important;border-color:#153f49!important}
button[data-variant="segmented_control"][aria-checked="true"] p {color:#fff!important}
[data-testid="stForm"] {border-color:#cbd2c6;background:#fdfbf5;border-radius:14px}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input {color:#153f49!important;background:#fffdf7!important}
[data-testid="stNumberInput"]>div,[data-baseweb="select"]>div {background:#fffdf7;color:#153f49;border-color:#a9bcb4}
[data-testid="stExpander"] summary {background:#eeeade;color:#153f49;border-radius:9px}
[data-testid="stExpander"] summary p {color:#153f49!important}
[data-testid="stDataFrame"] {border:1px solid #ccd4c9;border-radius:10px}
@media(max-width:800px){[data-testid="stMainBlockContainer"]{padding:4.5rem 1.15rem 3rem}.terrain-top{padding:14px;font-size:.62rem;letter-spacing:.06em}.result{padding:18px;gap:16px}.result .number{font-size:2.7rem}.mini{padding:10px}.mini strong{font-size:1.2rem}}
@media(max-width:420px){.mini-grid{grid-template-columns:1fr}.mini{display:flex;justify-content:space-between;align-items:center;gap:10px}.terrain-top span:last-child{display:none}.terrain-legend{padding:12px;gap:9px;font-size:.71rem}.result{display:block}.result .message{margin-top:14px}h1{font-size:2.2rem!important}}
</style>"""


def page_heading(kicker, title, subtitle):
    st.html(f'<div class="eyebrow">TALUSLAB · {html.escape(kicker)}</div>')
    st.title(title)
    st.html(f'<div class="intro">{html.escape(subtitle)}</div>')


def note(text):
    st.html(f'<div class="note">{html.escape(text)}</div>')


def metrics(items):
    st.html('<div class="mini-grid">' + ''.join(
        f'<div class="mini"><small>{html.escape(str(label))}</small><strong>{html.escape(str(value))}</strong></div>'
        for label, value in items) + '</div>')


def render_terrain(params):
    # st.html applique un profil de nettoyage HTML qui retire les SVG.
    # st.image affiche le SVG autonome en image sans fichier ni script externe.
    content = slope_svg(params)
    start, end = content.index('<svg'), content.index('</svg>')+6
    with st.container(border=True, key="terrain"):
        st.html('<div class="terrain-top"><span>COUPE DU TERRAIN</span><span>01 / MODÈLE MÉCANIQUE</span></div>')
        st.image(content[start:end], width="stretch", alt=f"Coupe de pente {params['beta']} degrés ; surface et plan parallèle, profondeur verticale {params['z']} m. Les flèches bleues représentent la pression u.")
        st.html('<div class="terrain-legend"><span><i class="line"></i>Surface du sol</span><span><i class="line dashed"></i>Plan étudié</span><span>z : verticale</span></div>')


def chart_style(chart):
    return chart.configure_view(stroke=None).configure_axis(
        labelColor=PETROL, titleColor=PETROL, gridColor="#DFE4DA", labelFontSize=12,
        titleFontSize=12).configure_legend(labelColor=PETROL, titleColor=PETROL).properties(background="#F7F5EE")


def init_state():
    defaults = {"params": copy.deepcopy(BASE), "mode": "Découverte", "scenarios": [],
                "soils": [], "simulation": None, "simulation_result": None,
                "revision": 0, "soil_revision": 0, "sim_revision": 0,
                "data_origin": "Valeurs initiales fictives", "report": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def activate_params(p, mode="Étude détaillée", origin="Saisie utilisateur"):
    st.session_state.params = validated_params(p)
    st.session_state.mode = mode
    st.session_state.data_origin = str(origin)[:1000]
    st.session_state.revision += 1
    st.session_state.report = None


def export_backup():
    return dict(application="TalusLab", version=VERSION, params=st.session_state.params,
                mode=st.session_state.mode, scenarios=st.session_state.scenarios,
                soils=st.session_state.soils, simulation=st.session_state.simulation,
                origin=st.session_state.data_origin)


def atelier():
    page_heading("Atelier 01 / Voir et comprendre", "Le terrain devient un calcul.",
                 "Faites varier la pente et le sol. La coupe, les efforts et le résultat évoluent ensemble.")
    mode = st.segmented_control("Niveau d’étude", ["Découverte", "Étude détaillée"],
                                default=st.session_state.mode, key=f"mode_{st.session_state.revision}",
                                selection_mode="single") or st.session_state.mode
    if mode != st.session_state.mode:
        p = copy.deepcopy(st.session_state.params)
        if mode == "Découverte":
            p.update(c=0., u=0.)
        activate_params(p, mode, st.session_state.data_origin)
        st.rerun()
    left, right = st.columns([1.65, 1], gap="large")
    with right:
        st.subheader("Réglages du terrain")
        rev, p = st.session_state.revision, st.session_state.params
        with st.form(f"terrain_{rev}"):
            q = {}
            q["beta"] = st.slider(LABELS["beta"], 1., 75., p["beta"], .5)
            q["phi"] = st.slider(LABELS["phi"], 0., 60., p["phi"], .5)
            if mode == "Étude détaillée":
                for key, step in (("c", .5), ("gamma", .5), ("z", .1), ("u", .5)):
                    q[key] = st.number_input(LABELS[key], *RANGES[key], value=p[key], step=step,
                                             help="Profondeur verticale du plan étudié, pas la hauteur totale du talus." if key == "z" else None)
            else:
                q.update(c=0., u=0., gamma=p["gamma"], z=p["z"])
                st.caption("Sol sans cohésion, u = 0. γ et z s’annulent dans la formule simplifiée.")
            apply = st.form_submit_button("Actualiser la coupe", type="primary", width="stretch", icon=":material/tune:")
        if apply:
            try:
                activate_params(q, mode)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
                st.caption("La coupe conserve le dernier calcul valide.")
        if st.button("Exemple fictif : pente 20°, frottement 30°", width="stretch"):
            activate_params(copy.deepcopy(BASE), "Découverte", "Exemple fictif — test 1")
            st.rerun()
        st.caption("Les réglages s’appliquent avec le bouton. Les bornes de saisie ne sont pas des seuils réglementaires.")
    p = st.session_state.params
    r = calculate(p)
    with left:
        render_terrain(p)
        message, color = interpretation(r["F"])
        st.html(f'<div class="result {color}"><div><div class="label">FACTEUR F · SANS UNITÉ</div>'
                f'<div class="number">{fmt(r["F"])}</div></div><div class="message">{message}</div></div>')
        st.caption("Origine : " + st.session_state.data_origin + ". Le seuil 1 correspond à l’équilibre du modèle.")
    metrics([("Contrainte normale σn", fmt(r["normal"]) + " kPa"),
             ("Effort moteur τ", fmt(r["shear"]) + " kPa"),
             ("Résistance disponible", fmt(r["strength"]) + " kPa")])
    with st.expander("Comprendre le calcul, étape par étape", expanded=mode == "Découverte"):
        if mode == "Découverte":
            st.latex(r"F=\frac{\tan\varphi'}{\tan\beta}")
            st.markdown(f"Avec φ′ = **{fmt(p['phi'],1)}°** et β = **{fmt(p['beta'],1)}°**, "
                        f"on obtient **F = {fmt(r['F'],4)}**. Plus la pente est raide, plus les efforts moteurs augmentent.")
        else:
            st.latex(r"\sigma_n=\gamma z\cos^2\beta,\quad\tau=\gamma z\sin\beta\cos\beta")
            st.latex(r"\sigma'_n=\sigma_n-u,\qquad F=\frac{c'+(\sigma_n-u)\tan\varphi'}{\tau}")
            st.markdown(f"La contrainte effective vaut **{fmt(r['effective'])} kPa**. "
                        "L’eau sous pression réduit cette contrainte et donc la résistance par frottement.")
    note(LIMITS)
    st.subheader("Garder ce scénario")
    with st.form("save_scenario"):
        name = st.text_input("Nom du scénario", max_chars=80, placeholder="Ex. Talus sec, pente 20°")
        save = st.form_submit_button("Ajouter à la comparaison", icon=":material/bookmark_add:")
    if save:
        name = name.strip()
        if not name:
            st.error("Donnez un nom au scénario.")
        elif name in [s["nom"] for s in st.session_state.scenarios]:
            st.error("Ce nom existe déjà. Choisissez un autre nom.")
        elif len(st.session_state.scenarios) >= 20:
            st.error("20 scénarios maximum. Retirez-en un dans Comparaison.")
        else:
            st.session_state.scenarios.append(dict(nom=name, params=copy.deepcopy(p), origine=st.session_state.data_origin))
            st.success("Scénario ajouté à la comparaison.")


def scenario_frame(scenarios):
    return pd.DataFrame([dict(Scénario=s["nom"], **s["params"], F=calculate(s["params"])["F"], Origine=s["origine"])
                         for s in scenarios], columns=["Scénario", *BASE, "F", "Origine"])


def comparison():
    page_heading("Atelier 02 / Comparer", "Une pente, plusieurs décisions.",
                 "Comparez des hypothèses explicites. Aucun traitement ne transforme automatiquement un terrain en ouvrage validé.")
    if st.button("Ajouter les 3 scénarios fictifs de démonstration", type="primary"):
        demos = [dict(nom="Démo — pente 20°, sèche", params=copy.deepcopy(BASE), origine="Fictif"),
                 dict(nom="Démo — pente 35°, sèche", params={**BASE, "beta": 35.}, origine="Fictif"),
                 dict(nom="Démo — pente 20°, u = 10 kPa", params={**BASE, "u": 10.}, origine="Fictif")]
        names = {s["nom"] for s in st.session_state.scenarios}
        additions = [s for s in demos if s["nom"] not in names]
        if len(st.session_state.scenarios)+len(additions) > 20:
            st.error("Libérez des places : la comparaison est limitée à 20 scénarios.")
        else:
            st.session_state.scenarios.extend(additions)
            st.success("Exemples ajoutés séparément. Vos scénarios sont conservés.")
    scenarios = st.session_state.scenarios
    if not scenarios:
        st.info("Enregistrez un scénario dans l’atelier, ou activez les exemples fictifs ci-dessus.")
        return
    frame = scenario_frame(scenarios)
    plot_frame = frame.assign(Repère=[f"S{i+1}" for i in range(len(frame))])
    bars = alt.Chart(plot_frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("F:Q", title="Facteur F (sans unité)"),
        y=alt.Y("Repère:N", sort=None, title=None),
        color=alt.condition("datum.F < 1", alt.value(RED), alt.value(PETROL)),
        tooltip=["Scénario:N", alt.Tooltip("F:Q", format=".3f"), "Origine:N"])
    line = alt.Chart(pd.DataFrame({"limite": [1]})).mark_rule(color=COPPER, strokeDash=[5, 4]).encode(x="limite:Q")
    st.altair_chart(chart_style((bars+line).properties(height=max(150, len(frame)*37))), width="stretch",
                    alt="Facteurs F des scénarios. La ligne à 1 représente l’équilibre du modèle.")
    st.caption(" · ".join(f"S{i+1} : {s['nom']}" for i,s in enumerate(scenarios)))
    st.dataframe(frame, hide_index=True, width="stretch", column_config={"F": st.column_config.NumberColumn(format="%.3f")},
                 alt="Hypothèses et résultats de chaque scénario sauvegardé.")
    choice = st.selectbox("Scénario à examiner", range(len(scenarios)), format_func=lambda i: scenarios[i]["nom"])
    chosen = scenarios[choice]
    st.caption("Scénario affiché : " + chosen["nom"])
    render_terrain(chosen["params"])
    a, b = st.columns(2)
    if a.button("Reprendre dans l’atelier", icon=":material/edit:", width="stretch"):
        activate_params(chosen["params"], "Étude détaillée", chosen["origine"])
        st.switch_page(st.session_state.pages[0])
    if b.button("Retirer ce scénario", icon=":material/delete:", width="stretch"):
        st.session_state.scenarios.pop(choice)
        st.rerun()
    note("La variation de u seule est une comparaison mécanique à γ constant. Une étude réelle de l’eau exige aussi "
         "de préciser l’état du sol, le poids volumique et les conditions hydrauliques.")


def soil_data():
    page_heading("Atelier 03 / Données", "Du carnet aux indicateurs.",
                 "Rassemblez des paramètres de sol comparables. Les mesures, les valeurs déclarées et les exemples fictifs restent identifiés.")
    if st.button("Ajouter les 7 essais fictifs", type="primary"):
        known = {str(r.get("id", "")) for r in st.session_state.soils}
        demo = example_soils()
        if len(st.session_state.soils)+len(demo) > 20000:
            st.error("Le carnet est limité à 20 000 lignes. Libérez de la place avant d’ajouter les exemples.")
        elif any(r["id"] in known for r in demo):
            st.warning("Les identifiants E01 à E07 sont déjà présents : aucun exemple ajouté.")
        else:
            st.session_state.soils.extend(demo)
            st.session_state.soil_revision += 1
            st.rerun()
    with st.expander("Saisir une observation"):
        with st.form("soil_manual", clear_on_submit=False):
            a, b = st.columns(2)
            ident = a.text_input("Identifiant de l’essai *", max_chars=160)
            serie = b.text_input("Série / même sol et protocole *", max_chars=160)
            phi = a.number_input("Frottement mesuré φ′ (°)", 0., 60., 30., .5)
            c = b.number_input("Cohésion mesurée c′ (kPa)", 0., 100., 0., .5)
            gamma = a.number_input("Poids volumique mesuré γ (kN/m³)", 10., 30., 20., .1)
            origin = b.selectbox("Origine de l’observation", ["Mesuré", "Déclaré", "Fictif"])
            add = st.form_submit_button("Enregistrer l’observation")
        if add:
            candidate = dict(id=ident, serie=serie, phi_deg=phi, c_kpa=c, gamma_kn_m3=gamma, origine=origin)
            if len(st.session_state.soils) >= 20000:
                st.error("Le carnet est limité à 20 000 lignes.")
            elif not ident.strip() or not serie.strip():
                st.error("Renseignez l’identifiant et la série.")
            elif ident.strip() in {str(r.get("id", "")).strip() for r in st.session_state.soils}:
                st.error("Cet identifiant existe déjà.")
            else:
                st.session_state.soils.append(candidate)
                st.session_state.soil_revision += 1
                st.success("Observation ajoutée.")
    with st.expander("Importer un CSV ou un classeur XLSX"):
        st.caption("Colonnes requises : phi_deg, c_kpa, gamma_kn_m3. Facultatives : id, serie, origine. "
                   "CSV UTF-8, virgule ou point-virgule ; première feuille Excel. Maximum 2 Mo / 20 000 lignes.")
        sample = pd.DataFrame(example_soils()).to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button("Modèle CSV fictif", sample, "TalusLab-exemple.csv", "text/csv")
        uploaded = st.file_uploader("Fichier de paramètres du sol", type=["csv", "xlsx"], key="soil_upload")
        if uploaded:
            try:
                raw = read_soil_file(uploaded.getvalue(), uploaded.name)
                good, errors = clean_soils(raw)
                st.info(f"{len(raw)} lignes ; {len(good)} analysables ; {len(errors)} exclues.")
                if not errors.empty:
                    st.dataframe(errors, hide_index=True, alt="Anomalies du fichier avant import.")
                replace = st.checkbox("Remplacer le carnet actif par ce fichier")
                if st.button("Utiliser ce fichier", disabled=not replace or good.empty):
                    st.session_state.soils = raw
                    st.session_state.soil_revision += 1
                    st.rerun()
            except Exception as exc:
                st.error("Lecture du fichier impossible. " + str(exc))
    if not st.session_state.soils:
        st.info("Le carnet est vide. Ajoutez une observation ou chargez les essais fictifs.")
        return
    with st.expander("Modifier ou supprimer des lignes du carnet"):
        with st.form("soil_edit"):
            edited = st.data_editor(pd.DataFrame(st.session_state.soils).astype(str), num_rows="dynamic",
                                    key=f"editor_{st.session_state.soil_revision}", width="stretch", hide_index=True,
                                    alt="Carnet modifiable. Les nombres et identifiants seront vérifiés à l’enregistrement.")
            update = st.form_submit_button("Appliquer les modifications du carnet")
        if update:
            try:
                records = edited.fillna("").to_dict("records")
                clean_soils(records)
                st.session_state.soils = records
                st.session_state.soil_revision += 1
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    good, errors = clean_soils(st.session_state.soils)
    metrics([("Lignes du carnet", str(len(st.session_state.soils))), ("Analysables", str(len(good))),
             ("Exclues et signalées", str(len(errors)))])
    if not errors.empty:
        st.warning("Les lignes ci-dessous restent dans le carnet brut et sont exclues de l’analyse. Aucune valeur n’est imputée.")
        st.dataframe(errors, hide_index=True, alt="Lignes exclues et raisons de l’exclusion.")
    if good.empty:
        return
    groups = sorted(good["serie"].unique())
    group = st.selectbox("Série à analyser", groups)
    selected = good[good.serie == group].copy()
    origins = sorted(selected.origine.unique())
    st.caption("Origine(s) dans cette série : " + ", ".join(origins) + ". Ne mélangez pas sols ou protocoles différents.")
    summary = soil_summary(selected)
    st.dataframe(summary, hide_index=True, width="stretch", alt="Moyenne, médiane, variance, écart-type et coefficient de variation par paramètre.",
                 column_config={col: st.column_config.NumberColumn(format="%.3f") for col in ["Moyenne", "Médiane", "Variance", "Écart-type", "CV (%)"]})
    st.caption("Variance et écart-type d’échantillon : diviseur n − 1, indisponibles pour n = 1. Variance dans l’unité au carré ; CV = 100 × écart-type / moyenne.")
    variable = st.selectbox("Variable du graphique", list(SOIL_NUM), format_func=SOIL_NAMES.get)
    hist = alt.Chart(selected).mark_bar(color=PETROL, cornerRadiusTopLeft=3).encode(
        x=alt.X(variable+":Q", bin=alt.Bin(maxbins=min(15, max(3, len(selected)))), title=SOIL_NAMES[variable]),
        y=alt.Y("count():Q", title="Nombre d’observations"), tooltip=[alt.Tooltip("count():Q", title="Effectif")])
    st.altair_chart(chart_style(hist.properties(height=240)), width="stretch", alt="Histogramme du paramètre sélectionné dans une seule série.")
    st.dataframe(selected, hide_index=True, width="stretch", alt="Observations analysables de la série sélectionnée.")
    if len(selected) < 10:
        note("Petit échantillon : les indicateurs décrivent ces lignes seulement. Ils ne prouvent ni une loi de probabilité, "
             "ni la représentativité du site. Les bornes de simulation doivent être justifiées séparément.")
    if st.button("Utiliser les moyennes de cette série dans l’atelier"):
        q = {**st.session_state.params, "phi": float(selected.phi_deg.mean()),
             "c": float(selected.c_kpa.mean()), "gamma": float(selected.gamma_kn_m3.mean())}
        try:
            activate_params(q, "Étude détaillée", "Moyennes de " + group + " / " + ", ".join(origins))
            st.success("Moyennes transférées. La pente, z et u sont conservés. Ouvrez l’atelier pour examiner le calcul.")
        except ValueError as exc:
            st.error(str(exc))


def uncertainties():
    page_heading("Atelier 04 / Incertitudes", "Que change un sol variable ?",
                 "Tirez des hypothèses de sol, recalculez F et comptez les cas F < 1. Le résultat dépend des lois choisies.")
    if st.button("Activer le test fictif : probabilité théorique 50 %", type="primary"):
        st.session_state.simulation = initial_sim()
        st.session_state.simulation_result = None
        st.session_state.sim_revision += 1
        st.rerun()
    cfg = st.session_state.simulation or initial_sim()
    note("Configuration indépendante de l’atelier : les paramètres fixes et les lois sont affichés ci-dessous. "
         "Le bouton de reprise permet de partir explicitement du terrain étudié.")
    if st.button("Reprendre le terrain de l’atelier pour la simulation"):
        p = copy.deepcopy(st.session_state.params)
        st.session_state.simulation = dict(base=p, bounds={k: [p[k], p[k]] for k in ("phi", "c", "u")},
                                           law="Uniforme", n=10000, seed=42)
        st.session_state.simulation_result = None
        st.session_state.sim_revision += 1
        st.rerun()
    with st.form(f"mc_{st.session_state.sim_revision}"):
        fixed = {}
        a, b, c = st.columns(3)
        fixed["beta"] = a.number_input("Pente fixe β (°)", 1., 75., cfg["base"]["beta"], .5)
        fixed["gamma"] = b.number_input("Poids volumique fixe γ (kN/m³)", 10., 30., cfg["base"]["gamma"], .5)
        fixed["z"] = c.number_input("Profondeur verticale fixe z (m)", .2, 20., cfg["base"]["z"], .1)
        law = st.selectbox("Loi des variables incertaines", ["Uniforme", "Triangulaire"], index=["Uniforme", "Triangulaire"].index(cfg["law"]))
        st.caption("Uniforme : toutes les valeurs de l’intervalle ont la même densité. Triangulaire : mode au milieu de l’intervalle. "
                   "Les trois variables sont supposées indépendantes. Deux bornes égales rendent une variable fixe.")
        bounds = {}
        for key, label in (("phi", "Frottement φ′ (°)"), ("c", "Cohésion c′ (kPa)"), ("u", "Pression u (kPa)")):
            l, r = st.columns(2)
            lo, hi = RANGES[key]
            bounds[key] = [l.number_input(label+" — minimum", lo, hi, cfg["bounds"][key][0], .5),
                           r.number_input(label+" — maximum", lo, hi, cfg["bounds"][key][1], .5)]
        n = st.select_slider("Nombre de tirages", options=[1000, 5000, 10000, 50000, 100000], value=cfg["n"] if cfg["n"] in [1000, 5000, 10000, 50000, 100000] else 10000)
        seed = st.number_input("Graine aléatoire (pour reproduire le test)", 0, 2147483647, cfg["seed"], 1)
        run = st.form_submit_button("Lancer les simulations", type="primary", icon=":material/play_arrow:")
    if run:
        try:
            q = {**fixed, **{k: (v[0]+v[1])/2 for k, v in bounds.items()}}
            candidate = validate_sim(dict(base=q, bounds=bounds, law=law, n=int(n), seed=int(seed)))
            with st.spinner("Calcul des tirages et des statistiques…"):
                res = simulate(candidate)
            st.session_state.simulation = candidate
            st.session_state.simulation_result = res
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.simulation_result = None
    result = st.session_state.simulation_result
    if result is None:
        st.info("Activez le test à 50 %, puis cliquez sur Lancer les simulations, ou définissez vos propres bornes.")
        return
    result_cfg = result["config"]
    st.subheader("Résultat de la dernière simulation exécutée")
    st.caption(f"β = {fmt(result_cfg['base']['beta'],1)}° ; γ = {fmt(result_cfg['base']['gamma'],1)} kN/m³ ; "
               f"z = {fmt(result_cfg['base']['z'],1)} m ; loi {result_cfg['law'].lower()} ; graine {result_cfg['seed']}. "
               "Modifier un champ du formulaire ne recalcule pas le résultat avant le lancement.")
    st.dataframe(pd.DataFrame([{"Variable": LABELS[k], "Minimum": v[0], "Maximum": v[1]} for k,v in result_cfg["bounds"].items()]),
                 hide_index=True, alt="Bornes réellement utilisées dans la dernière simulation.")
    metrics([("P(F < 1) sous ces hypothèses", fmt(100*result["probability"]) + " %"),
             ("Médiane de F", fmt(result["quantiles"][1],3)),
             ("Tirages F < 1", f"{result['count']} / {result['n']}")])
    st.caption(f"Intervalle de Wilson à 95 % lié aux seuls tirages : {fmt(100*result['ci'][0])} à {fmt(100*result['ci'][1])} %. "
               "Cet intervalle mesure l’erreur numérique Monte-Carlo ; il ne couvre pas les erreurs du modèle ni des données.")
    if result["exact"] is not None:
        st.success(f"Résultat théorique du cas sec uniforme : {fmt(100*result['exact'])} %. "
                   f"Écart Monte-Carlo : {fmt(100*(result['probability']-result['exact']))} point(s).")
    trials = result["trials"]
    counts, edges = np.histogram(trials.F, bins=40)
    hist_data = pd.DataFrame({"début": edges[:-1], "fin": edges[1:], "effectif": counts})
    bars = alt.Chart(hist_data).mark_bar(color=PETROL, orient="vertical").encode(
        x=alt.X("début:Q", title="F", scale=alt.Scale(zero=False)), x2="fin:Q",
        y=alt.Y("effectif:Q", title="Tirages", scale=alt.Scale(zero=True)),
        tooltip=["début", "fin", "effectif"])
    rule = alt.Chart(pd.DataFrame({"limite": [1]})).mark_rule(color=COPPER, strokeDash=[6,4]).encode(x="limite:Q")
    st.altair_chart(chart_style((bars+rule).properties(height=260)), width="stretch", alt="Histogramme simulé du facteur F, ligne d’équilibre à 1.")
    ordered = np.sort(trials.F.to_numpy())
    ids = np.unique(np.linspace(0, len(ordered)-1, min(1200, len(ordered))).astype(int))
    cdf = pd.DataFrame({"F": ordered[ids], "probabilité": (ids+1)/len(ordered)})
    st.altair_chart(chart_style(alt.Chart(cdf).mark_line(color=COPPER).encode(x="F:Q", y=alt.Y("probabilité:Q", title="Part des tirages ≤ F"), tooltip=[alt.Tooltip("F:Q", format=".3f"), alt.Tooltip("probabilité:Q", format=".2%")]).properties(height=220)),
                    width="stretch", alt="Distribution cumulée empirique des facteurs simulés.")
    st.markdown(f"**Quantiles de F** : 5 % = **{fmt(result['quantiles'][0],3)}** ; "
                f"50 % = **{fmt(result['quantiles'][1],3)}** ; 95 % = **{fmt(result['quantiles'][2],3)}**.")
    if result["sensitivity"]:
        sensitivity = pd.DataFrame(result["sensitivity"])
        sensitivity["Paramètre"] = sensitivity.variable.map({"phi_deg": "Frottement φ′", "c_kpa": "Cohésion c′", "u_kpa": "Pression u"})
        st.subheader("Sensibilité exploratoire")
        st.altair_chart(chart_style(alt.Chart(sensitivity).mark_bar(color=COPPER).encode(x=alt.X("rho:Q", scale=alt.Scale(domain=[-1,1]), title="Corrélation de rang de Spearman"), y=alt.Y("Paramètre:N", title=None), tooltip=["Paramètre", alt.Tooltip("rho:Q", format=".3f")]).properties(height=130)),
                        width="stretch", alt="Associations entre variables simulées et facteur F ; les variables fixes sont omises.")
        st.caption("Association dans cette simulation, dépendante des intervalles choisis. Elle ne prouve pas une causalité sur le terrain.")
    note("Cette probabilité décrit les hypothèses tirées au hasard, pas la probabilité annuelle d’un glissement réel. "
         "Aucun tirage n’est supprimé et aucune loi n’est ajustée automatiquement à un petit échantillon.")


def report_payload(title, author, include_sim):
    good, audit = clean_soils(st.session_state.soils)
    result = st.session_state.simulation_result if include_sim else None
    return dict(title=title.strip(), author=author.strip(), params=copy.deepcopy(st.session_state.params),
                mode=st.session_state.mode, origin=st.session_state.data_origin,
                scenarios=copy.deepcopy(st.session_state.scenarios), raw=copy.deepcopy(st.session_state.soils),
                good=good, audit=audit, simulation=result)


def payload_signature(payload):
    # Éviter de hacher les 100 000 tirages : ils découlent entièrement de la configuration.
    return signature({k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else
                           (v["config"] if k == "simulation" and v is not None else v))
                      for k, v in payload.items()})


def safe_excel(frame):
    """Échapper les textes interprétables comme formules par un tableur."""
    result = frame.copy()
    for col in result:
        result[col] = result[col].map(lambda x: "'"+x if isinstance(x, str) and x.startswith(("=", "+", "-", "@", "\t", "\r")) else x)
    return result


def csv_bytes(frame):
    return safe_excel(frame).to_csv(index=False, sep=";").encode("utf-8-sig")


def make_excel(payload):
    buffer = io.BytesIO()
    p, r = payload["params"], calculate(payload["params"])
    context = [{"Information": "Application", "Valeur": "TalusLab " + VERSION},
               {"Information": "Titre", "Valeur": payload["title"]},
               {"Information": "Auteur", "Valeur": payload["author"]},
               {"Information": "Origine", "Valeur": payload["origin"]},
               {"Information": "Limites", "Valeur": LIMITS}]
    frames = {"Contexte": pd.DataFrame(context),
              "Calcul actif": pd.DataFrame([dict(Paramètre=LABELS[k], Valeur=v) for k,v in p.items()] +
                                           [dict(Paramètre=k, Valeur=v) for k,v in r.items()]),
              "Scénarios": scenario_frame(payload["scenarios"]),
              "Carnet brut": pd.DataFrame(payload["raw"]), "Sols analysables": payload["good"],
              "Exclusions": payload["audit"]}
    if not payload["good"].empty:
        frames["Statistiques par série"] = pd.concat([soil_summary(g).assign(Série=name) for name,g in payload["good"].groupby("serie")], ignore_index=True)
    sim = payload["simulation"]
    if sim is not None:
        frames["Hypothèses MC"] = pd.DataFrame([dict(Paramètre=k, Valeur=json.dumps(v, ensure_ascii=False)) for k,v in sim["config"].items()])
        frames["Résultat MC"] = pd.DataFrame([dict(P_F_inf_1=sim["probability"], n=sim["n"], cas=sim["count"],
                                                 IC_MC_bas=sim["ci"][0], IC_MC_haut=sim["ci"][1],
                                                 quantile_05=sim["quantiles"][0], médiane=sim["quantiles"][1], quantile_95=sim["quantiles"][2])])
        frames["Tirages MC"] = sim["trials"]
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in frames.items():
            safe_excel(frame).to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            from openpyxl.styles import Alignment, Font, PatternFill
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="153F49")
                cell.font = Font(color="FFFFFF", bold=True)
            from openpyxl.utils import get_column_letter
            for i, col in enumerate(frame.columns, 1):
                lengths = [len(str(col))]+[len(str(x)) for x in frame[col].head(80)]
                sheet.column_dimensions[get_column_letter(i)].width = min(60, max(15, max(lengths)+2))
            for row in sheet.iter_rows(min_row=2, max_row=min(sheet.max_row, 500)):
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return buffer.getvalue()


def make_pdf(payload):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.graphics.shapes import Drawing, Polygon, Line, String, Rect
    import reportlab
    fontdir = Path(reportlab.__file__).parent / "fonts"
    # Polices fournies par ReportLab : aucun fichier à ajouter au dossier livré.
    if "TalusVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("TalusVera", str(fontdir/"Vera.ttf")))
        pdfmetrics.registerFont(TTFont("TalusVeraBold", str(fontdir/"VeraBd.ttf")))
        pdfmetrics.registerFontFamily("TalusVera", normal="TalusVera", bold="TalusVeraBold", italic="TalusVera", boldItalic="TalusVeraBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TLBody", fontName="TalusVera", fontSize=9, leading=14,
                             textColor=colors.HexColor(PETROL), spaceAfter=7, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="TLTitle", parent=styles["TLBody"], fontName="TalusVeraBold", fontSize=25, leading=31, spaceAfter=16))
    styles.add(ParagraphStyle(name="TLHead", parent=styles["TLBody"], fontName="TalusVeraBold", fontSize=13, leading=18,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="TLCell", parent=styles["TLBody"], fontSize=8, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle(name="TLFormula", parent=styles["TLBody"], fontSize=10, leading=16, backColor=colors.HexColor("#EFE9DB"), borderPadding=10, spaceBefore=8, spaceAfter=14))
    def esc(value):
        # Vera est livrée avec ReportLab mais ne possède pas les primes et tous
        # les caractères grecs. Transcrire ces symboles préserve leur sens.
        replacements = {"′": "'", "β": "beta", "φ": "phi", "γ": "gamma", "σ": "sigma", "τ": "tau", "Σ": "somme"}
        text = str(value)
        for old,new in replacements.items():
            text = text.replace(old,new)
        return html.escape(text)
    P = lambda x, style="TLBody": Paragraph(esc(x), styles[style])
    def table(rows, widths):
        cells = [[P(v, "TLCell") for v in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DFE8DF")),
                               ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 8),
                               ("LEFTPADDING", (0,0), (-1,-1), 9),
                               ("LINEBELOW", (0,0), (-1,-1), .4, colors.HexColor("#D5DED4"))]))
        return t
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=51, bottomMargin=49, title=payload["title"], author=payload["author"] or "TalusLab")
    p, r = payload["params"], calculate(payload["params"])
    story = [P("TALUSLAB / CARNET D’ÉTUDE", "TLHead"), P(payload["title"], "TLTitle"),
             P("Étudiant / groupe : " + (payload["author"] or "Non renseigné")),
             P("Édité le " + datetime.now().strftime("%d/%m/%Y à %H:%M") + " — version " + VERSION),
             P("Origine du calcul actif : " + payload["origin"]),
             P(f"Facteur F = {fmt(r['F'],4)} — {interpretation(r['F'])[0]}.", "TLHead")]
    g = slope_geometry(p)
    d = Drawing(500, 275)
    d.add(Rect(0,0,500,275,fillColor=colors.HexColor("#F0F2E9"),strokeColor=None))
    project = lambda pt: (19+pt[0]*.66, 275-pt[1]*.66)
    d.add(Polygon([v for point in g["polygon"] for v in project(point)], fillColor=colors.HexColor(SAND), strokeColor=None))
    for key, col, dash in (("surface", PETROL, None), ("plane", COPPER, [6,4])):
        a,b = map(project, g[key])
        d.add(Line(*a,*b,strokeColor=colors.HexColor(col),strokeWidth=2,strokeDashArray=dash))
    (ax, ay), (bx, by) = g["surface"]
    cy = g["plane"][0][1]
    a, b = project((ax-15,ay)), project((ax-15,cy))
    d.add(Line(*a,*b,strokeColor=colors.HexColor(PETROL),strokeWidth=1))
    d.add(String(a[0]-13,(a[1]+b[1])/2,"z",fontName="TalusVera",fontSize=11,fillColor=colors.HexColor(PETROL)))
    d.add(String(16,250,f"Pente = {fmt(p['beta'],1)}° ; frottement = {fmt(p['phi'],1)}°",fontName="TalusVera",fontSize=11,fillColor=colors.HexColor(PETROL)))
    d.add(String(16,13,"Surface : trait plein. Plan étudié : pointillés. z : verticale. Échelles x/y égales.",fontName="TalusVera",fontSize=8,fillColor=colors.HexColor(PETROL)))
    story += [Spacer(1,8), d, Spacer(1,10), P(LIMITS), PageBreak(), P("1 / Hypothèses et calcul vérifiable", "TLHead")]
    labels_ascii = dict(beta="Pente (°)", phi="Frottement effectif (°)", c="Cohésion effective (kPa)",
                        gamma="Poids volumique (kN/m³)", z="Profondeur verticale (m)", u="Pression interstitielle (kPa)")
    story.append(table([["Paramètre", "Valeur"]]+[[labels_ascii[k], fmt(v,3)] for k,v in p.items()], [350,159]))
    story += [P("sigma_n = gamma × z × cos²(beta) ; tau = gamma × z × sin(beta) × cos(beta)", "TLFormula"),
              P("F = [c + (sigma_n − u) × tan(phi)] / tau", "TLFormula"),
              P(f"Contrainte normale : {fmt(r['normal'],4)} kPa ; contrainte effective : {fmt(r['effective'],4)} kPa. "
                f"Effort moteur : {fmt(r['shear'],4)} kPa ; résistance : {fmt(r['strength'],4)} kPa."),
              P("La pression u doit être comprise entre 0 et sigma_n. Au-delà, ce modèle ne traite pas la perte de contact. "
                "Les bornes de saisie sont des limites pédagogiques de l’interface, pas des normes.")]
    if payload["scenarios"]:
        story.append(P("2 / Scénarios conservés", "TLHead"))
        rows = [["Scénario / origine", "Pente / frottement", "c / u (kPa)", "F"]]
        for s in payload["scenarios"]:
            sp = s["params"]
            rows.append([s["nom"]+" / "+s["origine"], f"{fmt(sp['beta'],1)}° / {fmt(sp['phi'],1)}°",
                         f"{fmt(sp['c'],1)} / {fmt(sp['u'],1)}", fmt(calculate(sp)["F"],4)])
        story.append(table(rows, [234,113,95,67]))
        story.append(P("Les paramètres complets de chaque scénario figurent dans le classeur Excel et le carnet JSON."))
    else:
        story.append(P("2 / Scénarios : aucun scénario conservé", "TLHead"))
    story.append(P("3 / Carnet de sol et qualité des données", "TLHead"))
    story.append(P(f"{len(payload['raw'])} lignes brutes ; {len(payload['good'])} analysables ; {len(payload['audit'])} exclues. "
                   "Aucune imputation. Les séries sont traitées séparément. Les lignes détaillées et les exclusions sont exportées en Excel."))
    if not payload["good"].empty:
        for name, group in payload["good"].groupby("serie"):
            rows = [["Variable / unité", "Moyenne", "Écart-type", "CV (%)"]]
            for _, row in soil_summary(group).iterrows():
                # Libellés sans glyphes grecs pour la police embarquée.
                label = row["Variable"].replace("φ′", "phi").replace("γ", "gamma")
                rows.append([label, fmt(row["Moyenne"],3), fmt(row["Écart-type"],3), fmt(row["CV (%)"],2)])
            story.append(KeepTogether([P(f"Série : {name} ; n = {len(group)} ; origine(s) : {', '.join(sorted(group.origine.unique()))}"), table(rows,[245,88,88,88])]))
        story.append(P("Écart-type d’échantillon : diviseur n − 1. Petit échantillon : descriptif uniquement ; représentativité et loi non établies."))
    sim = payload["simulation"]
    if sim is not None:
        cfg = sim["config"]
        story += [PageBreak(), P("4 / Simulation indépendante du calcul actif", "TLHead"),
                  P(f"P(F < 1) = {fmt(100*sim['probability'])} % ; {sim['count']} cas sur {sim['n']} tirages. "
                    f"Loi : {cfg['law']} ; graine : {cfg['seed']}. Tous les tirages sont conservés."),
                  P(f"Fixes : pente {fmt(cfg['base']['beta'],1)}° ; poids volumique {fmt(cfg['base']['gamma'],1)} kN/m³ ; "
                    f"profondeur verticale {fmt(cfg['base']['z'],1)} m."),
                  table([["Variable", "Minimum", "Maximum"]]+[[labels_ascii[k],fmt(v[0],3),fmt(v[1],3)] for k,v in cfg["bounds"].items()], [289,110,110]),
                  P("Les variables sont indépendantes. Pour une loi triangulaire, le mode est au milieu de l’intervalle. "
                    "Les valeurs identiques aux deux bornes sont des constantes."),
                  P(f"Quantiles de F : 5 % = {fmt(sim['quantiles'][0],4)} ; médiane = {fmt(sim['quantiles'][1],4)} ; "
                    f"95 % = {fmt(sim['quantiles'][2],4)}."),
                  P(f"Intervalle de Wilson à 95 % : {fmt(100*sim['ci'][0])} à {fmt(100*sim['ci'][1])} %. "
                    "Il porte seulement sur l’erreur de tirage, pas sur l’incertitude réelle du terrain.")]
        if sim["exact"] is not None:
            story.append(P(f"Cas sec uniforme : P(phi < beta) théorique = {fmt(100*sim['exact'])} %."))
        counts, edges = np.histogram(sim["trials"].F, bins=30)
        drawing = Drawing(500,200)
        x0,x1,y0,y1 = 45,480,35,175
        maximum = max(1,int(max(counts)))
        for i,count in enumerate(counts):
            x = x0+(x1-x0)*i/len(counts)
            drawing.add(Rect(x,y0,(x1-x0)/len(counts)-1,(y1-y0)*count/maximum,fillColor=colors.HexColor(PETROL),strokeColor=None))
        drawing.add(Line(x0,y0,x1,y0,strokeColor=colors.HexColor(PETROL)))
        drawing.add(String(x0,17,fmt(edges[0],3),fontName="TalusVera",fontSize=8))
        drawing.add(String(x1-25,17,fmt(edges[-1],3),fontName="TalusVera",fontSize=8))
        drawing.add(String(155,187,"Histogramme simulé de F — effectifs",fontName="TalusVera",fontSize=9))
        if edges[0] <= 1 <= edges[-1]:
            xx=x0+(1-edges[0])/(edges[-1]-edges[0])*(x1-x0)
            drawing.add(Line(xx,y0,xx,y1,strokeColor=colors.HexColor(COPPER),strokeWidth=2))
        story += [drawing, P("Cette probabilité dépend des lois et intervalles choisis. Ce n’est pas une probabilité annuelle de glissement "
                                           "ni une attestation de sécurité. Les tirages complets sont dans Excel.")]
    story += [PageBreak(), P("5 / Trois tests pour comprendre et vérifier", "TLHead"),
              P("TEST 1 — Sol sec sans cohésion : pente 20°, frottement 30°, c = 0, u = 0. "
                "F = tan(30°) / tan(20°) = 1,5863. À 35° de pente, F = 0,8245. Calculatrice en degrés."),
              P("TEST 2 — Eau sous pression : pente 30°, frottement 30°, gamma = 20 kN/m³, z = 2 m, c = 0, u = 10 kPa. "
                "sigma_n = 30 kPa ; tau = 17,3205 kPa ; F = 0,6667. Avec u = 0, F = 1."),
              P("TEST 3 — Incertitude : pente 30°, c = u = 0, phi uniforme entre 25° et 35°. "
                "F < 1 revient à phi < 30°. La moitié de l’intervalle convient : probabilité théorique 50 %. "
                "Pour 10 000 tirages, l’erreur typique à 95 % vaut environ ±0,98 point, sans garantie pour une réalisation particulière."),
              P("Sources et portée", "TLHead"), P(LIMITS)]
    for title,url in SOURCES:
        story.append(KeepTogether([P(title), Paragraph(f'<link href="{esc(url)}" color="#153F49">{esc(url)}</link>', styles["TLBody"])]))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CAD5CA")); canvas.line(43,36,552,36)
        canvas.setFont("TalusVera",8); canvas.setFillColor(colors.HexColor(PETROL))
        canvas.drawString(43,23,"TalusLab · modèle pédagogique · " + VERSION)
        canvas.drawRightString(552,23,str(doc.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def reports():
    page_heading("Atelier 05 / Expliquer et transmettre", "Un résultat qui se défend.",
                 "Exportez les hypothèses avec les résultats. Sauvegardez le carnet pour retrouver votre travail plus tard.")
    title = st.text_input("Titre du rapport", value="Étude pédagogique d’une pente", max_chars=160,
                          key="report_title", persist_state="session")
    author = st.text_input("Étudiant / groupe (facultatif)", max_chars=160, key="report_author", persist_state="session")
    include_sim = st.checkbox("Inclure la dernière simulation exécutée", disabled=st.session_state.simulation_result is None,
                              key="report_include_sim", persist_state="session")
    payload = report_payload(title, author, include_sim and st.session_state.simulation_result is not None)
    sig = payload_signature(payload)
    if st.button("Préparer le PDF et Excel", type="primary", icon=":material/picture_as_pdf:"):
        if not title.strip():
            st.error("Donnez un titre au rapport.")
        else:
            with st.spinner("Préparation du carnet d’étude…"):
                st.session_state.report = dict(signature=sig, pdf=make_pdf(payload), excel=make_excel(payload))
    report = st.session_state.report
    if report and report["signature"] == sig:
        a,b = st.columns(2)
        a.download_button("Télécharger le PDF", report["pdf"], "TalusLab-rapport.pdf", "application/pdf", width="stretch")
        b.download_button("Télécharger Excel", report["excel"], "TalusLab-analyse.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    elif report:
        st.caption("Les données ou le titre ont changé : préparez à nouveau les fichiers pour obtenir un rapport à jour.")
    a,b = st.columns(2)
    backup = json.dumps(export_backup(), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    a.download_button("Sauvegarder le carnet JSON", backup, "TalusLab-carnet.json", "application/json", width="stretch")
    good, audit = clean_soils(st.session_state.soils)
    b.download_button("Données de sol CSV", csv_bytes(good), "TalusLab-sols.csv", "text/csv", width="stretch")
    if st.session_state.simulation_result is not None:
        st.download_button("Tirages de simulation CSV", csv_bytes(st.session_state.simulation_result["trials"]),
                           "TalusLab-tirages.csv", "text/csv")
    st.caption("Le JSON conserve les paramètres, les scénarios, les lignes brutes et la configuration de simulation. "
               "Les tirages sont reproduits avec la graine. Aucun stockage permanent sur le serveur.")
    with st.expander("Restaurer un carnet sauvegardé"):
        upload = st.file_uploader("Carnet TalusLab JSON", type="json", key="restore_upload")
        if upload:
            try:
                if upload.size > 20_000_000:
                    raise ValueError("Sauvegarde limitée à 20 Mo.")
                def reject_constant(value):
                    raise ValueError("Les valeurs JSON NaN et infinies ne sont pas acceptées.")
                restored = validate_backup(json.loads(upload.getvalue().decode("utf-8-sig"), parse_constant=reject_constant))
                st.caption(f"{len(restored['scenarios'])} scénario(s), {len(restored['soils'])} ligne(s) de sol.")
                confirm = st.checkbox("Remplacer le carnet et ses réglages par cette sauvegarde")
                if st.button("Restaurer le carnet", disabled=not confirm):
                    activate_params(restored["params"], restored["mode"], restored["origin"])
                    for key in ("scenarios", "soils", "simulation"):
                        st.session_state[key] = restored[key]
                    st.session_state.soil_revision += 1
                    st.session_state.sim_revision += 1
                    st.session_state.simulation_result = simulate(restored["simulation"]) if restored["simulation"] else None
                    st.rerun()
            except Exception as exc:
                st.error("Sauvegarde non restaurée : " + str(exc))
    st.divider()
    st.subheader("Parcours de démonstration")
    st.markdown("""1. **Atelier du talus** : activez l’exemple fictif. β = 20°, φ′ = 30° donnent **F = 1,5863**.
2. **Comparaison** : ajoutez les trois scénarios. La pente 35° donne **F = 0,8245** ; à 20° avec u = 10 kPa, **F = 1,1372**.
3. **Données de sol** : ajoutez les sept essais. Moyennes : **φ′ = 30°**, **c′ = 4 kPa**, **γ = 19,5 kN/m³**. Écart-type de φ′ : **2,1602°**.
4. **Incertitudes** : activez le test à 50 %, puis lancez la simulation. Avec 10 000 tirages, le résultat se situe généralement à environ **50 % ± 1 point**.
5. **Rapports** : incluez la simulation, préparez les fichiers et sauvegardez le carnet JSON.""")
    with st.expander("Les hypothèses à expliquer devant l’enseignant"):
        st.markdown("""- **F** compare la résistance disponible aux efforts moteurs sur un plan donné.
- **Pente infinie** : on représente une fenêtre d’une pente longue et homogène ; les effets des extrémités sont négligés.
- **z** est la profondeur verticale entre la surface et le plan parallèle étudié.
- **c′ et φ′** sont des paramètres en contraintes effectives ; cette application ne traite pas la résistance non drainée.
- **u** est une pression interstitielle au plan étudié. Aucune conversion automatique pluie → pression n’est proposée.
- **Monte-Carlo** répète le calcul avec des valeurs tirées au hasard. Les lois et l’indépendance sont des hypothèses à justifier.
- **F = 1** est l’équilibre théorique du modèle. Les couleurs ne constituent pas une validation de sécurité.
- **Les observations** sont décrites par série. La variabilité observée et les intervalles de simulation ne sont pas automatiquement équivalents.""")
    note(LIMITS)
    for label,url in SOURCES:
        st.markdown(f"[{label}]({url})")


def main():
    st.set_page_config(page_title="TalusLab · Atelier des pentes", page_icon=":material/landscape:", layout="wide")
    st.html(CSS)
    init_state()
    pages = [st.Page(atelier, title="Atelier du talus", icon=":material/landscape:", default=True),
             st.Page(comparison, title="Comparaison", icon=":material/compare_arrows:", url_path="comparaison"),
             st.Page(soil_data, title="Données de sol", icon=":material/table_chart:", url_path="donnees"),
             st.Page(uncertainties, title="Incertitudes", icon=":material/analytics:", url_path="incertitudes"),
             st.Page(reports, title="Rapports et méthodes", icon=":material/description:", url_path="rapports")]
    st.session_state.pages = pages
    with st.sidebar:
        st.html('<div class="brand">talus<b>lab.</b></div><div class="side-note">LIRE LE TERRAIN<br>COMPRENDRE LES INCERTITUDES</div>')
    navigation = st.navigation(pages)
    with st.sidebar:
        st.divider()
        st.markdown("**Un modèle, des hypothèses.**")
        st.caption("Atelier pédagogique · Génie civil\n\nPente infinie / contraintes effectives")
        st.markdown(f"**{len(st.session_state.scenarios)}** scénario(s) conservé(s)")
        st.caption("Conservez votre travail avec le carnet JSON dans Rapports et méthodes. Stockage limité à la session.")
    navigation.run()


if __name__ == "__main__":
    main()
