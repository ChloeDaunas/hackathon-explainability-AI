"""
PIPELINE - Axe Éthique : Audit Fairness + Explainabilité (SHAP + LIME)
========================================================================
Entrée  : data/HR_Dataset_Anonymized.csv  (sorti par anonymize_data.py)
          models/rf_model.joblib           (sorti par train_model.py)
Sorties : data/fairness_audit_results.csv
          data/shap_feature_importance.csv

IMPORTANT : ce script charge le modèle sauvegardé par train_model.py.
Sex, RaceDesc ET GenderID sont exclus des features du modèle,
mais conservés ici pour mesurer les biais a posteriori.
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Chargement données ──────────────────────────────────────────────────────
df_raw = pd.read_csv("data/HR_Dataset_Anonymized.csv")
print(f"Dataset chargé : {df_raw.shape[0]} employés")

# Conserver les attributs sensibles AVANT tout encodage
df_sensitive = df_raw[["Sex", "RaceDesc"]].copy()
df_sensitive["Sex"]      = df_sensitive["Sex"].str.strip()
df_sensitive["RaceDesc"] = df_sensitive["RaceDesc"].str.strip()


# ── 2. Même preprocessing que train_model.py (obligatoire pour aligner les features) ──
df = df_raw.copy()
for col in ["DateofHire", "DateofTermination", "LastPerformanceReview_Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

reference_cutoff = df["LastPerformanceReview_Date"].dropna().max()
df["TenureYears"] = ((reference_cutoff - df["DateofHire"]).dt.days / 365.25).round(2).clip(lower=0)

# Même liste d'exclusions que train_model.py (post-event + dates + IDs)
# PLUS : GenderID exclu car encode le genre (variable sensible numérique)
cols_to_drop = [
    "DateofTermination", "TermReason", "EmploymentStatus", "EmpStatusID",
    "DateofHire", "LastPerformanceReview_Date", "EmpID",
]
# Attributs sensibles exclus du modèle — conservés uniquement pour l'audit
fairness_cols = ["Sex", "RaceDesc", "GenderID"]

cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop_existing)

y = df["Termd"].astype(int)
X_raw = df.drop(columns=["Termd"] + [c for c in fairness_cols if c in df.columns])

categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)

print(f"Features du modèle : {X.shape[1]} colonnes")
print(f"Attributs sensibles exclus : {[c for c in fairness_cols if c in df_raw.columns]}")


# ── 3. Chargement du modèle sauvegardé ────────────────────────────────────────
rf_model = joblib.load("models/rf_model.joblib")
with open("models/model_feature_columns.json") as f:
    saved_columns = json.load(f)

# Aligner les colonnes avec celles utilisées à l'entraînement
X_aligned = X.reindex(columns=saved_columns, fill_value=0)
print(f"Colonnes modèle : {len(saved_columns)} | Colonnes actuelles : {X.shape[1]}")

# Reproduire le même split (random_state=42 + stratify) pour récupérer le même test set
X_train, X_test, y_train, y_test = train_test_split(
    X_aligned, y, test_size=0.20, random_state=42, stratify=y
)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
test_idx = X_test.index

print("\n=== Performance du modèle (rappel) ===")
print(classification_report(y_test, y_pred, target_names=["Active", "Left"], zero_division=0))


# ── 4. Audit Fairness — métriques manuelles ───────────────────────────────────
def fairness_metrics(df_audit, sensitive_col, privileged_value):
    priv   = df_audit[df_audit[sensitive_col] == privileged_value]
    unpriv = df_audit[df_audit[sensitive_col] != privileged_value]

    def metrics(group):
        tp = ((group["y_pred"] == 1) & (group["y_true"] == 1)).sum()
        fp = ((group["y_pred"] == 1) & (group["y_true"] == 0)).sum()
        tn = ((group["y_pred"] == 0) & (group["y_true"] == 0)).sum()
        fn = ((group["y_pred"] == 0) & (group["y_true"] == 1)).sum()
        n  = len(group)
        return {
            "n": n,
            "pos_rate": (tp + fp) / n if n > 0 else 0,
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        }

    p, u = metrics(priv), metrics(unpriv)
    return {
        "privileged": privileged_value,
        "priv_n": p["n"],   "priv_pos_rate": round(p["pos_rate"], 4),
        "unpriv_n": u["n"], "unpriv_pos_rate": round(u["pos_rate"], 4),
        "demographic_parity_diff": round(u["pos_rate"] - p["pos_rate"], 4),
        "disparate_impact":        round(u["pos_rate"] / p["pos_rate"], 4) if p["pos_rate"] > 0 else float("nan"),
        "equalized_odds_diff":     round(max(abs(u["TPR"] - p["TPR"]), abs(u["FPR"] - p["FPR"])), 4),
    }

# Dataframe de test enrichi avec attributs sensibles
df_audit = pd.DataFrame({
    "Sex":     df_sensitive.loc[test_idx, "Sex"].values,
    "RaceDesc": df_sensitive.loc[test_idx, "RaceDesc"].values,
    "y_true":  y_test.values,
    "y_pred":  y_pred,
    "y_prob":  y_prob,
}, index=test_idx)

gender_result = fairness_metrics(df_audit, "Sex",      privileged_value="M")
race_result   = fairness_metrics(df_audit, "RaceDesc", privileged_value="White")

print("\n=== Audit Fairness — Genre ===")
print(f"  Demographic Parity Diff : {gender_result['demographic_parity_diff']:+.4f}  (idéal = 0)")
print(f"  Disparate Impact        : {gender_result['disparate_impact']:.4f}  (acceptable : 0.8–1.25)")
print(f"  Equalized Odds Diff     : {gender_result['equalized_odds_diff']:.4f}  (idéal = 0)")

print("\n=== Audit Fairness — Ethnie ===")
print(f"  Demographic Parity Diff : {race_result['demographic_parity_diff']:+.4f}  (idéal = 0)")
print(f"  Disparate Impact        : {race_result['disparate_impact']:.4f}  (acceptable : 0.8–1.25)")
print(f"  Equalized Odds Diff     : {race_result['equalized_odds_diff']:.4f}  (idéal = 0)")


# ── 5. AIF360 (si installé) ───────────────────────────────────────────────────
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    df_aif = df_audit.copy()
    df_aif["sex_num"] = (df_aif["Sex"] == "M").astype(int)  # 1 = M (privilégié)

    aif_true = BinaryLabelDataset(
        df=df_aif[["y_true", "sex_num"]].rename(columns={"y_true": "Termd"}),
        label_names=["Termd"],
        protected_attribute_names=["sex_num"],
        favorable_label=0,    # Active = résultat favorable pour l'employé
        unfavorable_label=1,
    )
    aif_pred = aif_true.copy()
    aif_pred.labels = df_aif["y_pred"].values.reshape(-1, 1)

    clf_metric = ClassificationMetric(
        aif_true, aif_pred,
        privileged_groups=[{"sex_num": 1}],
        unprivileged_groups=[{"sex_num": 0}],
    )
    print("\n=== AIF360 — Genre ===")
    print(f"  Statistical Parity Difference : {clf_metric.statistical_parity_difference():.4f}")
    print(f"  Disparate Impact              : {clf_metric.disparate_impact():.4f}")
    print(f"  Equal Opportunity Difference  : {clf_metric.equal_opportunity_difference():.4f}")
except ImportError:
    print("\n[AIF360 non installé] Les métriques manuelles ci-dessus sont équivalentes.")
    print("  Pour installer : pip install aif360")


# ── 6. SHAP — Explicabilité globale + locale ──────────────────────────────────
import shap

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Compatibilité SHAP ancien (<0.41) et nouveau (>=0.41)
shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
print(f"\nSHAP values calculées : {shap_class1.shape}")

# Graphique global (bar)
plt.figure()
shap.summary_plot(shap_class1, X_test, max_display=15, plot_type="bar", show=False)
plt.title("Importance globale des features (SHAP)")
plt.tight_layout()
plt.savefig("data/shap_global.png", dpi=100, bbox_inches="tight")
print("✓ data/shap_global.png sauvegardé")

# Explication locale : employé le plus à risque
i = int(np.argmax(y_prob))
base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

shap_exp = shap.Explanation(
    values=shap_class1[i],
    base_values=base_val,
    data=X_test.iloc[i].values,
    feature_names=list(X_test.columns),
)
plt.figure()
shap.plots.waterfall(shap_exp, max_display=12, show=False)
plt.title(f"Pourquoi l'employé #{X_test.index[i]} est prédit démissionnaire")
plt.tight_layout()
plt.savefig("data/shap_local.png", dpi=100, bbox_inches="tight")
print(f"✓ data/shap_local.png sauvegardé (employé #{X_test.index[i]}, prob={y_prob[i]:.1%})")


# ── 7. Sauvegarde des résultats ───────────────────────────────────────────────
summary_df = pd.DataFrame([
    {"Attribut": "Sex",      "Groupe privilégié": "M (Homme)",   "Dem. Parity Diff": gender_result["demographic_parity_diff"], "Disparate Impact": gender_result["disparate_impact"], "Eq. Odds Diff": gender_result["equalized_odds_diff"]},
    {"Attribut": "RaceDesc", "Groupe privilégié": "White",       "Dem. Parity Diff": race_result["demographic_parity_diff"],   "Disparate Impact": race_result["disparate_impact"],   "Eq. Odds Diff": race_result["equalized_odds_diff"]},
])
summary_df.to_csv("data/fairness_audit_results.csv", index=False)

shap_df = pd.DataFrame(shap_class1, columns=X_test.columns)
shap_df.abs().mean().sort_values(ascending=False).reset_index() \
    .rename(columns={"index": "Feature", 0: "Mean_SHAP"}) \
    .to_csv("data/shap_feature_importance.csv", index=False)

print("\n✓ data/fairness_audit_results.csv sauvegardé")
print("✓ data/shap_feature_importance.csv sauvegardé")
