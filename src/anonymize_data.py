"""
PIPELINE - Axe Cybersécurité : Anonymisation RGPD
===================================================
Techniques démontrées :
  1. Suppression    — colonnes directement identifiantes (nom, email...)
  2. Généralisation — DOB → tranche d'âge (utile pour le ML)
  3. Pseudonymisation — EmpID remplacé par un ID aléatoire reproductible
  4. Masquage       — Zip réduit aux 2 premiers chiffres
  5. Justification  — Sex & RaceDesc conservés explicitement pour l'audit fairness
"""

import pandas as pd
import hashlib
from datetime import date

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/HRDataset_v14.csv")
print(f"Dataset chargé : {df.shape[0]} employés, {df.shape[1]} colonnes")


# ── 2. SUPPRESSION — données directement identifiantes ────────────────────────
cols_to_drop = ["Employee_Name", "ManagerName", "ManagerID"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
print(f"[Suppression] Colonnes supprimées : {cols_to_drop}")


# ── 3. PSEUDONYMISATION — EmpID remplacé par un hash SHA-256 tronqué ──────────
# Reproductible (même input → même hash) mais non réversible
if "EmpID" in df.columns:
    df["EmpID"] = df["EmpID"].astype(str).apply(
        lambda x: "EMP-" + hashlib.sha256(x.encode()).hexdigest()[:8].upper()
    )
    print("[Pseudonymisation] EmpID remplacé par des codes anonymes")


# ── 4. GÉNÉRALISATION — DOB → tranche d'âge ────────────────────────────────────
# On garde l'info utile pour le ML sans exposer la date exacte
if "DOB" in df.columns:
    today = date.today()
    df["Age"] = pd.to_datetime(df["DOB"], errors="coerce").apply(
        lambda d: today.year - d.year if pd.notna(d) else None
    )
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 29, 39, 49, 59, 100],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"]
    )
    df = df.drop(columns=["DOB", "Age"])
    print("[Généralisation] DOB → tranche d'âge (AgeGroup)")


# ── 5. MASQUAGE — Zip réduit aux 2 premiers chiffres ─────────────────────────
if "Zip" in df.columns:
    df["Zip"] = df["Zip"].astype(str).str[:2] + "***"
    print("[Masquage] Zip → 2 premiers chiffres seulement")


# ── 6. Justification explicite des colonnes sensibles conservées ──────────────
# Sex et RaceDesc sont volontairement CONSERVÉS pour l'audit fairness (axe Éthique).
# Ils ne seront PAS utilisés comme features dans le modèle ML,
# uniquement pour mesurer les biais a posteriori.
print("\n[RGPD Note] Sex & RaceDesc conservés pour audit fairness uniquement.")
print("  → Ne seront pas des features du modèle prédictif.")


# ── 7. Vérification finale ────────────────────────────────────────────────────
print(f"\nDataset anonymisé : {df.shape[0]} employés, {df.shape[1]} colonnes")
print("Colonnes restantes :", df.columns.tolist())


# ── 8. Sauvegarde ─────────────────────────────────────────────────────────────
df.to_csv("data/HR_Dataset_Anonymized.csv", index=False)
print("\n✓ Fichier sauvegardé : data/HR_Dataset_Anonymized.csv")
