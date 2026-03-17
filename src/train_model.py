"""
PIPELINE - Modèle ML : Prédiction du risque de démission
==========================================================
Entrée  : data/HR_Dataset_Anonymized.csv  (sorti par anonymize_data.py)
Sorties : models/rf_model.joblib
          models/model_feature_columns.json
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/HR_Dataset_Anonymized.csv")
print(f"Dataset chargé : {df.shape[0]} employés, {df.shape[1]} colonnes")


# ── 2. Nettoyage dates ─────────────────────────────────────────────────────────
for col in ["DateofHire", "DateofTermination", "LastPerformanceReview_Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ── 3. Variable cible ──────────────────────────────────────────────────────────
# Termd : 0 = encore en poste | 1 = a quitté
print(f"\nTaux de turnover global : {df['Termd'].mean():.1%}")


# ── 4. Feature engineering (sans fuite de données) ────────────────────────────
model_df = df.copy()

# Tenure calculée avec une date de référence globale (pas de fuite post-événement)
reference_cutoff = model_df["LastPerformanceReview_Date"].dropna().max()
model_df["TenureYears"] = ((reference_cutoff - model_df["DateofHire"]).dt.days / 365.25).round(2).clip(lower=0)

# Colonnes à supprimer : post-événement (fuites), dates brutes, IDs
cols_to_drop = [
    "DateofTermination", "TermReason", "EmploymentStatus", "EmpStatusID",  # post-événement
    "DateofHire", "LastPerformanceReview_Date",                             # dates brutes
    "EmpID",                                                                 # ID pseudonymisé
]
# IMPORTANT : Sex et RaceDesc exclus du modèle (gardés uniquement pour l'audit fairness)
fairness_cols = ["Sex", "RaceDesc","GenderID"]

cols_to_drop_existing = [c for c in cols_to_drop if c in model_df.columns]
model_df = model_df.drop(columns=cols_to_drop_existing)

print(f"Colonnes supprimées (fuite/dates/IDs) : {cols_to_drop_existing}")
print(f"Colonnes fairness exclues du modèle   : {[c for c in fairness_cols if c in model_df.columns]}")

# Séparation features / target (Sex et RaceDesc mis de côté)
y = model_df["Termd"].astype(int)
X_raw = model_df.drop(columns=["Termd"] + [c for c in fairness_cols if c in model_df.columns])

# One-hot encoding des variables catégorielles
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)

print(f"\nShape X : {X.shape} | Shape y : {y.shape}")


# ── 5. Entraînement Random Forest ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# n_estimators=100 : largement suffisant pour ~400 lignes
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n=== Évaluation du modèle ===")
print(classification_report(y_test, y_pred, target_names=["Active", "Left"], zero_division=0))
print("Matrice de confusion [[TN FP] [FN TP]] :")
print(confusion_matrix(y_test, y_pred))


# ── 6. Feature importance (Top 15) ────────────────────────────────────────────
importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

top15 = importance_df.head(15)
print(f"\n=== Top 15 features ===")
print(top15.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(top15["Feature"][::-1], top15["Importance"][::-1], color="teal")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=100)
plt.show()


# ── 7. Sauvegarde ─────────────────────────────────────────────────────────────
import os; os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.joblib")
with open("models/model_feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n✓ Modèle sauvegardé : models/rf_model.joblib")
print("✓ Feature columns   : models/model_feature_columns.json")
print(f"  (référence tenure : {pd.Timestamp(reference_cutoff).date()})")
