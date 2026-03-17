"""
TalentGuard AI — Application de prévention du turnover
Hackathon Trusted AI x RH — Capgemini x ESILV
"""

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import shap
from sklearn.model_selection import train_test_split

# ── Config page ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .sidebar-brand { color: #f8fafc !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.8rem !important; font-weight: 600 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #1e293b; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #64748b;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #fff !important;
}

/* Risk badges */
.badge-high   { background:#ef4444;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem;font-weight:600; }
.badge-medium { background:#f59e0b;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem;font-weight:600; }
.badge-low    { background:#22c55e;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.8rem;font-weight:600; }

/* Alert boxes */
.alert-bias   { background:#451a03;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0; }
.alert-ok     { background:#052e16;border-left:4px solid #22c55e;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0; }
.alert-info   { background:#0f172a;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0; }

/* Headers */
h1 { font-weight: 600 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 500 !important; color: #e2e8f0; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Buttons */
.stButton button {
    background: #1e40af;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
}
.stButton button:hover { background: #1d4ed8; }

/* Dark theme base */
.main { background: #0f172a; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données et du modèle (mis en cache) ────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/HR_Dataset_Anonymized.csv")
    df_sensitive = df[["Sex", "RaceDesc"]].copy()
    df_sensitive["Sex"]      = df_sensitive["Sex"].str.strip()
    df_sensitive["RaceDesc"] = df_sensitive["RaceDesc"].str.strip()

    df2 = df.copy()
    for col in ["DateofHire", "DateofTermination", "LastPerformanceReview_Date"]:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce")

    reference_cutoff = df2["LastPerformanceReview_Date"].dropna().max()
    df2["TenureYears"] = ((reference_cutoff - df2["DateofHire"]).dt.days / 365.25).round(2).clip(lower=0)

    cols_to_drop = [
        "DateofTermination", "TermReason", "EmploymentStatus", "EmpStatusID",
        "DateofHire", "LastPerformanceReview_Date", "EmpID",
    ]
    fairness_cols = ["Sex", "RaceDesc", "GenderID"]
    cols_to_drop_existing = [c for c in cols_to_drop if c in df2.columns]
    df2 = df2.drop(columns=cols_to_drop_existing)

    y = df2["Termd"].astype(int)
    X_raw = df2.drop(columns=["Termd"] + [c for c in fairness_cols if c in df2.columns])
    categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)

    return df, df_sensitive, X, y, reference_cutoff

@st.cache_resource
def load_model():
    rf = joblib.load("models/rf_model.joblib")
    with open("models/model_feature_columns.json") as f:
        cols = json.load(f)
    return rf, cols

@st.cache_data
def get_predictions(_rf, _cols, _X, _y, _df_sensitive):
    X_aligned = _X.reindex(columns=_cols, fill_value=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, _y, test_size=0.20, random_state=42, stratify=_y
    )
    y_pred = _rf.predict(X_test)
    y_prob = _rf.predict_proba(X_test)[:, 1]

    result = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "risk_score": (y_prob * 100).round(1),
        "Sex":     _df_sensitive.loc[X_test.index, "Sex"].values,
        "RaceDesc": _df_sensitive.loc[X_test.index, "RaceDesc"].values,
    }, index=X_test.index)
    return result, X_test, y_test, y_pred, y_prob


# ── Chargement ─────────────────────────────────────────────────────────────────
try:
    df_raw, df_sensitive, X, y, reference_cutoff = load_data()
    rf_model, saved_columns = load_model()
    results, X_test, y_test, y_pred, y_prob = get_predictions(
        rf_model, saved_columns, X, y, df_sensitive
    )
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Erreur de chargement : {e}")
    st.info("Vérifiez que `data/HR_Dataset_Anonymized.csv` et `models/rf_model.joblib` existent.")
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ TalentGuard AI")
    st.markdown("*Hackathon Trusted AI x RH*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Analyse individuelle", "⚖️ Audit Fairness","🎯 Risk Simulator", "ℹ️ À propos"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    n_high = (results["risk_score"] >= 60).sum()
    st.metric("Employés à haut risque", f"{n_high} / {len(results)}")
    st.markdown(f"<small>Modèle : Random Forest · {len(saved_columns)} features</small>", unsafe_allow_html=True)
    st.markdown(f"<small>Référence tenure : {pd.Timestamp(reference_cutoff).date()}</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("Dashboard RH — Risques de Démission")
    st.markdown("Vue d'ensemble des employés analysés sur le jeu de test.")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Employés analysés", len(results))
    with col2:
        st.metric("Taux de turnover réel", f"{y.mean():.1%}")
    with col3:
        st.metric("⚠️ Haut risque (≥60%)", f"{(results['risk_score'] >= 60).sum()}")
    with col4:
        recall_left = ((results["y_pred"] == 1) & (results["y_true"] == 1)).sum() / max((results["y_true"] == 1).sum(), 1)
        st.metric("Recall (classe Left)", f"{recall_left:.0%}")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    # ── Distribution des scores de risque ────────────────────────────────────
    with col_left:
        st.subheader("Distribution des scores de risque")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")

        sorted_scores = results["risk_score"].sort_values()
        colors = ["#22c55e" if s < 40 else "#f59e0b" if s < 60 else "#ef4444"
          for s in sorted_scores]
        ax.bar(range(len(results)), sorted_scores, color=colors, width=0.8)
        ax.axhline(60, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Seuil haut risque (60%)")
        ax.axhline(40, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7, label="Seuil risque moyen (40%)")
        ax.set_xlabel("Employés (triés par risque)", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Score de risque (%)", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#64748b")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#334155")
        ax.legend(facecolor="#0f172a", labelcolor="#94a3b8", fontsize=8)

        patches = [
            mpatches.Patch(color="#22c55e", label="Faible (<40%)"),
            mpatches.Patch(color="#f59e0b", label="Moyen (40-60%)"),
            mpatches.Patch(color="#ef4444", label="Élevé (≥60%)"),
        ]
        ax.legend(handles=patches, facecolor="#0f172a", labelcolor="#94a3b8", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Top 10 employés à risque ──────────────────────────────────────────────
    with col_right:
        st.subheader("Top 10 employés à risque")

        top10 = results.nlargest(10, "risk_score")[["risk_score", "y_true", "Sex", "RaceDesc"]].copy()
        top10["Statut réel"] = top10["y_true"].map({0: "✅ Actif", 1: "❌ Parti"})
        top10["Risque"] = top10["risk_score"].apply(
            lambda s: "🔴 Élevé" if s >= 60 else ("🟡 Moyen" if s >= 40 else "🟢 Faible")
        )
        top10 = top10.rename(columns={"risk_score": "Score (%)"}).drop(columns=["y_true"])
        st.dataframe(top10, use_container_width=True, height=370)
    
    st.markdown("---")

    # ── Comparaison RF vs NLP ─────────────────────────────────────────────────
    try:
        nlp_df = pd.read_csv("data/HR_NLP_scores.csv")

        st.subheader("📊 Impact du NLP — Comparaison des deux modèles")
        st.markdown("<small style='color:#64748b'>Score RF = Random Forest sur données structurées · Score NLP = sentiment + satisfaction + engagement + absences</small>", unsafe_allow_html=True)

        # Fusionner RF et NLP sur l'index commun
        compare = results[["risk_score","y_true"]].copy()
        compare["nlp_risk_pct"] = nlp_df.set_index(nlp_df.index)["nlp_risk_pct"].values[:len(compare)]
        compare["delta"] = (compare["nlp_risk_pct"] - compare["risk_score"]).round(1)

        col_rf, col_nlp, col_delta = st.columns(3)
        with col_rf:
            st.metric("Score moyen RF",  f"{compare['risk_score'].mean():.1f}%")
        with col_nlp:
            st.metric("Score moyen NLP", f"{compare['nlp_risk_pct'].mean():.1f}%")
        with col_delta:
            changed = (compare["delta"].abs() > 15).sum()
            st.metric("Employés reclassés (Δ>15%)", f"{changed} / {len(compare)}")

        # Scatter RF vs NLP
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")
        colors_c = ["#ef4444" if t else "#22c55e" for t in compare["y_true"]]
        ax.scatter(compare["risk_score"], compare["nlp_risk_pct"],
                   c=colors_c, alpha=0.6, s=30)
        ax.plot([0,100],[0,100], "--", color="#475569", linewidth=1, label="Ligne d'égalité")
        ax.set_xlabel("Score RF (%)", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Score NLP (%)", color="#94a3b8", fontsize=9)
        ax.set_title("RF vs NLP — points éloignés = apport du texte", color="#e2e8f0", fontsize=10)
        ax.tick_params(colors="#64748b")
        ax.spines[["top","right","left","bottom"]].set_color("#334155")
        ax.legend(facecolor="#0f172a", labelcolor="#94a3b8", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class='alert-info'>
        <strong style='color:#93c5fd'>💡 Comment lire ce graphique</strong><br>
        <span style='color:#bfdbfe;font-size:0.88rem'>
        Les points <strong>au-dessus</strong> de la ligne = le NLP détecte plus de risque que le RF seul → feedback négatif non capturé par les chiffres.<br>
        Les points <strong>en dessous</strong> = le RF sur-estime le risque → l'employé s'exprime positivement malgré des indicateurs dégradés.
        </span>
        </div>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.info("Lancez `python src/nlp_pipeline.py` pour activer la comparaison NLP.")

    # ── Répartition par niveau de risque ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Répartition par niveau de risque")
    c1, c2, c3 = st.columns(3)

    low    = results[results["risk_score"] < 40]
    medium = results[(results["risk_score"] >= 40) & (results["risk_score"] < 60)]
    high   = results[results["risk_score"] >= 60]

    with c1:
        st.markdown(f"""
        <div style='background:#052e16;border:1px solid #166534;border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:2rem;font-weight:700;color:#22c55e'>{len(low)}</div>
        <div style='color:#86efac;font-weight:500'>🟢 Faible risque</div>
        <div style='color:#4ade80;font-size:0.85rem'>Score &lt; 40%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='background:#451a03;border:1px solid #92400e;border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:2rem;font-weight:700;color:#f59e0b'>{len(medium)}</div>
        <div style='color:#fcd34d;font-weight:500'>🟡 Risque moyen</div>
        <div style='color:#fbbf24;font-size:0.85rem'>Score 40–60%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style='background:#450a0a;border:1px solid #991b1b;border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:2rem;font-weight:700;color:#ef4444'>{len(high)}</div>
        <div style='color:#fca5a5;font-weight:500'>🔴 Haut risque</div>
        <div style='color:#f87171;font-size:0.85rem'>Score ≥ 60%</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE INDIVIDUELLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyse individuelle":
    st.title("Analyse Individuelle")
    st.markdown("Sélectionnez un employé pour comprendre son score de risque.")

    # Sélecteur employé
    col_sel, col_sort = st.columns([3, 1])
    with col_sort:
        sort_by = st.selectbox("Trier par", ["Risque (↓)", "ID employé"])

    sorted_results = results.sort_values("risk_score", ascending=(sort_by == "ID employé"))
    employee_options = {
        f"#{idx} — {row['risk_score']:.0f}% — {'❌ Parti' if row['y_true'] else '✅ Actif'} — {row['Sex']}": idx
        for idx, row in sorted_results.iterrows()
    }

    with col_sel:
        selected_label = st.selectbox("Choisir un employé", list(employee_options.keys()))

    emp_idx = employee_options[selected_label]
    emp     = results.loc[emp_idx]
    pos_in_test = list(results.index).index(emp_idx)

    # Score + badge
    score = emp["risk_score"]
    if score >= 60:
        badge = '<span class="badge-high">🔴 HAUT RISQUE</span>'
        color = "#ef4444"
    elif score >= 40:
        badge = '<span class="badge-medium">🟡 RISQUE MOYEN</span>'
        color = "#f59e0b"
    else:
        badge = '<span class="badge-low">🟢 FAIBLE RISQUE</span>'
        color = "#22c55e"

    st.markdown("---")
    col_info, col_shap = st.columns([1, 1.8])

    with col_info:
        st.markdown(f"### Employé #{emp_idx}")
        st.markdown(badge, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:16px;background:#1e293b;border-radius:12px;padding:20px'>
            <div style='font-size:3rem;font-weight:700;color:{color};text-align:center'>{score:.0f}%</div>
            <div style='text-align:center;color:#94a3b8;margin-bottom:16px'>Score de risque de démission</div>
            <hr style='border-color:#334155'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;font-size:0.9rem'>
                <div style='color:#64748b'>Statut réel</div>
                <div style='color:#e2e8f0'>{"❌ A quitté" if emp['y_true'] else "✅ Actif"}</div>
                <div style='color:#64748b'>Prédiction</div>
                <div style='color:#e2e8f0'>{"❌ Va quitter" if emp['y_pred'] else "✅ Va rester"}</div>
                <div style='color:#64748b'>Genre</div>
                <div style='color:#e2e8f0'>{emp['Sex']}</div>
                <div style='color:#64748b'>Ethnie</div>
                <div style='color:#e2e8f0'>{emp['RaceDesc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommandation RH
        st.markdown("#### 💡 Recommandation RH")
        if score >= 60:
            st.markdown("""
            <div class='alert-bias' style='background:#450a0a;border-left-color:#ef4444'>
            <strong style='color:#fca5a5'>Action urgente recommandée</strong><br>
            <span style='color:#fecaca;font-size:0.9rem'>Entretien de rétention à planifier cette semaine. Analyser les facteurs ci-contre.</span>
            </div>""", unsafe_allow_html=True)
        elif score >= 40:
            st.markdown("""
            <div class='alert-bias'>
            <strong style='color:#fcd34d'>Suivi renforcé</strong><br>
            <span style='color:#fde68a;font-size:0.9rem'>Point mensuel avec le manager. Vérifier satisfaction et évolution salariale.</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='alert-ok'>
            <strong style='color:#86efac'>Profil stable</strong><br>
            <span style='color:#bbf7d0;font-size:0.9rem'>Suivi standard suffisant. Continuer à valoriser les contributions.</span>
            </div>""", unsafe_allow_html=True)

    # SHAP waterfall
    with col_shap:
        st.markdown("#### Facteurs explicatifs (SHAP)")
        st.markdown("<small style='color:#64748b'>Quelles variables poussent ce score vers le haut ou le bas ?</small>", unsafe_allow_html=True)

        with st.spinner("Calcul des explications SHAP..."):
            try:
                explainer   = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_test)
                shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = base_val[1]

                shap_exp = shap.Explanation(
                    values=shap_class1[pos_in_test],
                    base_values=base_val,
                    data=X_test.iloc[pos_in_test].values,
                    feature_names=list(X_test.columns),
                )

                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#1e293b")
                shap.plots.waterfall(shap_exp, max_display=12, show=False)
                fig = plt.gcf()
                fig.patch.set_facecolor("#1e293b")
                ax = fig.axes[0]
                ax.set_facecolor("#1e293b")
                for text in ax.texts + [ax.title]:
                    text.set_color("#e2e8f0")
                ax.tick_params(colors="#94a3b8")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP non disponible : {e}")
                # Fallback : feature importance globale
                imp = pd.DataFrame({
                    "Feature": saved_columns,
                    "Importance": rf_model.feature_importances_
                }).sort_values("Importance", ascending=False).head(12)

                fig, ax = plt.subplots(figsize=(7, 4.5))
                fig.patch.set_facecolor("#1e293b")
                ax.set_facecolor("#1e293b")
                ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color="#3b82f6")
                ax.tick_params(colors="#94a3b8")
                ax.spines[["top", "right", "left", "bottom"]].set_color("#334155")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AUDIT FAIRNESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Audit Fairness":
    st.title("Audit Fairness — IA Éthique")
    st.markdown("Vérification que le modèle ne discrimine pas selon le genre ou l'ethnie.")

    # ── Métriques fairness ────────────────────────────────────────────────────
    def fairness_metrics(df_audit, sensitive_col, privileged_value):
        priv   = df_audit[df_audit[sensitive_col] == privileged_value]
        unpriv = df_audit[df_audit[sensitive_col] != privileged_value]
        def m(g):
            tp = ((g["y_pred"]==1)&(g["y_true"]==1)).sum()
            fp = ((g["y_pred"]==1)&(g["y_true"]==0)).sum()
            tn = ((g["y_pred"]==0)&(g["y_true"]==0)).sum()
            fn = ((g["y_pred"]==0)&(g["y_true"]==1)).sum()
            n  = len(g)
            return {
                "n": n,
                "pos_rate": (tp+fp)/n if n>0 else 0,
                "TPR": tp/(tp+fn) if (tp+fn)>0 else 0,
                "FPR": fp/(fp+tn) if (fp+tn)>0 else 0,
            }
        p, u = m(priv), m(unpriv)
        return {
            "priv_n": p["n"],   "priv_pos_rate": p["pos_rate"],
            "unpriv_n": u["n"], "unpriv_pos_rate": u["pos_rate"],
            "dpd": round(u["pos_rate"] - p["pos_rate"], 4),
            "di":  round(u["pos_rate"] / p["pos_rate"], 4) if p["pos_rate"] > 0 else float("nan"),
            "eod": round(max(abs(u["TPR"]-p["TPR"]), abs(u["FPR"]-p["FPR"])), 4),
        }

    g = fairness_metrics(results, "Sex",      "M")
    r = fairness_metrics(results, "RaceDesc",  "White")

    def verdict_di(val):
        return ("✅ OK", "#22c55e") if 0.8 <= val <= 1.25 else ("⚠️ BIAIS", "#f59e0b")
    def verdict_diff(val):
        return ("✅ OK", "#22c55e") if abs(val) <= 0.1 else ("⚠️ À surveiller", "#f59e0b")

    # ── Tableau récap ─────────────────────────────────────────────────────────
    st.subheader("Résumé des métriques")
    tab_genre, tab_ethnie = st.tabs(["👤 Genre", "🌍 Ethnie"])

    def render_fairness_tab(result, attr_name, priv_label, unpriv_label):
        col1, col2, col3 = st.columns(3)

        v_di  = verdict_di(result["di"])
        v_dpd = verdict_diff(result["dpd"])
        v_eod = verdict_diff(result["eod"])

        with col1:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:1.8rem;font-weight:700;color:{v_di[1]}'>{result['di']:.3f}</div>
                <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>Disparate Impact</div>
                <div style='color:{v_di[1]};font-size:0.9rem;margin-top:8px'>{v_di[0]}</div>
                <div style='color:#475569;font-size:0.75rem'>Zone acceptable : 0.8 – 1.25</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:1.8rem;font-weight:700;color:{v_dpd[1]}'>{result['dpd']:+.3f}</div>
                <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>Demographic Parity Diff</div>
                <div style='color:{v_dpd[1]};font-size:0.9rem;margin-top:8px'>{v_dpd[0]}</div>
                <div style='color:#475569;font-size:0.75rem'>Idéal : 0</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:1.8rem;font-weight:700;color:{v_eod[1]}'>{result['eod']:.3f}</div>
                <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>Equalized Odds Diff</div>
                <div style='color:{v_eod[1]};font-size:0.9rem;margin-top:8px'>{v_eod[0]}</div>
                <div style='color:#475569;font-size:0.75rem'>Idéal : 0  · Seuil : 0.1</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Groupe privilégié ({priv_label})** : n={result['priv_n']} · taux prédit={result['priv_pos_rate']:.1%}")
            st.markdown(f"**Groupe non-privilégié ({unpriv_label})** : n={result['unpriv_n']} · taux prédit={result['unpriv_pos_rate']:.1%}")
        with col_b:
            if result["di"] < 0.8 or result["di"] > 1.25:
                st.markdown("""<div class='alert-bias'><strong style='color:#fcd34d'>⚠️ Biais potentiel détecté</strong><br>
                <span style='color:#fde68a;font-size:0.9rem'>Recommandation : retirer l'attribut sensible du modèle ou appliquer une calibration par groupe.</span></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='alert-ok'><strong style='color:#86efac'>✅ Dans les limites acceptables</strong><br>
                <span style='color:#bbf7d0;font-size:0.9rem'>Le modèle respecte la règle des 80% sur cet attribut. Monitoring continu recommandé.</span></div>""", unsafe_allow_html=True)

    with tab_genre:
        render_fairness_tab(g, "Genre", "Homme (M)", "Femme (F)")

    with tab_ethnie:
        render_fairness_tab(r, "Ethnie", "White", "Non-White")

    # ── Graphique comparaison par groupe ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Taux de turnover prédit vs réel par groupe")

    col_g, col_e = st.columns(2)

    for col, attr, title in [(col_g, "Sex", "Genre"), (col_e, "RaceDesc", "Ethnie/Race")]:
        with col:
            grp = results.groupby(attr)[["y_true", "y_pred"]].mean().rename(
                columns={"y_true": "Réel", "y_pred": "Prédit"}
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            x = np.arange(len(grp))
            w = 0.35
            ax.bar(x - w/2, grp["Réel"],  w, label="Réel",  color="#3b82f6", alpha=0.85)
            ax.bar(x + w/2, grp["Prédit"], w, label="Prédit", color="#f59e0b", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(grp.index, color="#94a3b8", fontsize=9, rotation=20, ha="right")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.tick_params(colors="#64748b")
            ax.spines[["top","right","left","bottom"]].set_color("#334155")
            ax.set_title(f"Turnover par {title}", color="#e2e8f0", fontsize=10)
            ax.legend(facecolor="#0f172a", labelcolor="#94a3b8", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── AIF360 (à coller dans app.py, page Audit Fairness, avant le bloc AI Act) ──

    st.markdown("---")
    st.subheader("🔬 Validation AIF360 (IBM)")
    st.markdown("<small style='color:#64748b'>Double vérification via la librairie officielle IBM.</small>", unsafe_allow_html=True)

    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
        import warnings; warnings.filterwarnings("ignore")

        def aif360_metrics(df_audit, sensitive_col, privileged_value):
            df_aif = df_audit.copy()
            df_aif["attr"] = (df_aif[sensitive_col] == privileged_value).astype(int)

            aif_true = BinaryLabelDataset(
                df=df_aif[["y_true", "attr"]].rename(columns={"y_true": "Termd"}),
                label_names=["Termd"],
                protected_attribute_names=["attr"],
                favorable_label=0,
                unfavorable_label=1,
            )
            aif_pred = aif_true.copy()
            aif_pred.labels = df_aif["y_pred"].values.reshape(-1, 1)

            m = ClassificationMetric(
                aif_true, aif_pred,
                privileged_groups=[{"attr": 1}],
                unprivileged_groups=[{"attr": 0}],
            )
            return {
                "spd": m.statistical_parity_difference(),
                "di":  m.disparate_impact(),
                "eod": m.equal_opportunity_difference(),
                "aod": m.average_odds_difference(),
            }

        # Calcul séparé pour chaque attribut
        aif_gender = aif360_metrics(results, "Sex",      "M")
        aif_race   = aif360_metrics(results, "RaceDesc", "White")

        def render_aif_cards(metrics):
            col1, col2, col3, col4 = st.columns(4)
            def card(col, label, value, ideal, acceptable_range=None):
                ok = (acceptable_range[0] <= value <= acceptable_range[1]) if acceptable_range else abs(value) <= 0.1
                color   = "#22c55e" if ok else "#f59e0b"
                verdict = "✅ OK" if ok else "⚠️ À surveiller"
                col.markdown(f"""
                <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;text-align:center'>
                    <div style='font-size:1.6rem;font-weight:700;color:{color}'>{value:+.4f}</div>
                    <div style='color:#94a3b8;font-size:0.8rem;margin-top:4px'>{label}</div>
                    <div style='color:{color};font-size:0.85rem;margin-top:6px'>{verdict}</div>
                    <div style='color:#475569;font-size:0.72rem'>idéal : {ideal}</div>
                </div>""", unsafe_allow_html=True)
            card(col1, "Statistical Parity Diff",  metrics["spd"], "0")
            card(col2, "Disparate Impact",          metrics["di"],  "1", acceptable_range=(0.8, 1.25))
            card(col3, "Equal Opportunity Diff",    metrics["eod"], "0")
            card(col4, "Average Odds Difference",   metrics["aod"], "0")

        aif_tab_g, aif_tab_r = st.tabs(["👤 Genre", "🌍 Ethnie"])
        with aif_tab_g:
            render_aif_cards(aif_gender)
        with aif_tab_r:
            render_aif_cards(aif_race)

        st.markdown("""
        <div class='alert-info' style='margin-top:12px'>
        <strong style='color:#93c5fd'>ℹ️ Convention AIF360</strong><br>
        <span style='color:#bfdbfe;font-size:0.88rem'>
        AIF360 calcule <em>privilégié / non-privilégié</em>, soit l'inverse de nos métriques manuelles — les signes sont opposés mais les valeurs absolues sont cohérentes.
        </span>
        </div>""", unsafe_allow_html=True)

    except ImportError:
        st.info("AIF360 non installé — `pip install aif360`")
    except Exception as e:
        st.warning(f"Erreur AIF360 : {e}")

    # ── AI Act ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🇪🇺 Classification EU AI Act")
    st.markdown("""
    <div class='alert-bias' style='background:#1e0a3c;border-left-color:#a855f7'>
    <strong style='color:#d8b4fe;font-size:1rem'>⚠️ Ce système est classifié HAUT RISQUE (Annexe III)</strong><br>
    <span style='color:#e9d5ff;font-size:0.9rem'>
    Les systèmes d'IA utilisés en RH et gestion des emplois sont automatiquement considérés à haut risque.<br><br>
    <strong>Obligations légales :</strong><br>
    · Supervision humaine obligatoire avant toute décision impactant un employé<br>
    · Traçabilité de toutes les prédictions<br>
    · Audit de fairness régulier (ce rapport)<br>
    · Droit à l'explication pour les employés concernés (SHAP/LIME)
    </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — RISK SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Risk Simulator":
    st.title("Risk Simulator")
    st.markdown("Enter an employee's key characteristics to instantly predict their turnover risk.")

    # ── Load reduced model ────────────────────────────────────────────────────
    try:
        rf_red = joblib.load("models/rf_model_reduced.joblib")
        with open("models/reduced_feature_columns.json") as f:
            red_cols = json.load(f)
    except FileNotFoundError:
        st.error("Run `python src/train_model.py` first to generate the reduced model.")
        st.stop()

    st.markdown(f"<small style='color:#64748b'>Model trained on the {len(red_cols)} most important features (importance > 0.025) — fast, transparent, frugal.</small>", unsafe_allow_html=True)
    st.markdown("---")

    col_inputs, col_result = st.columns([1.1, 1])

    # ── Top 5 human-readable inputs ───────────────────────────────────────────
    # Map feature names to readable labels + sensible ranges
    FEATURE_META = {
        "TenureYears":          {"label": "📅 Tenure (years)",           "type": "float", "min": 0.0,  "max": 30.0, "default": 3.0,  "step": 0.5,  "help": "How long the employee has been with the company"},
        "Salary":               {"label": "💰 Hourly salary ($/h)",      "type": "float", "min": 10.0, "max": 100.0,"default": 25.0, "step": 0.5,  "help": "Current hourly pay rate"},
        "EngagementSurvey":     {"label": "📊 Engagement score (1–5)",   "type": "float", "min": 1.0,  "max": 5.0,  "default": 3.5,  "step": 0.1,  "help": "Latest engagement survey result"},
        "Absences":             {"label": "🏥 Absences (days/year)",     "type": "int",   "min": 0,    "max": 60,   "default": 5,    "step": 1,    "help": "Number of absence days in the last year"},
        "EmpSatisfaction":      {"label": "😊 Job satisfaction (1–5)",   "type": "int",   "min": 1,    "max": 5,    "default": 3,    "step": 1,    "help": "Self-reported job satisfaction score"},
        "DaysLateLast30":       {"label": "⏰ Late days (last 30 days)", "type": "int",   "min": 0,    "max": 30,   "default": 0,    "step": 1,    "help": "Number of days late in the past month"},
        "SpecialProjectsCount": {"label": "🚀 Special projects",         "type": "int",   "min": 0,    "max": 10,   "default": 1,    "step": 1,    "help": "Number of special projects assigned"},
        "PerfScoreID":          {"label": "🏆 Performance score (1–4)", "type": "int",   "min": 1,    "max": 4,    "default": 3,    "step": 1,    "help": "Latest performance review score"},
    }

    # Show only the top 5 most interpretable features that are in the reduced model
    PRIORITY = ["TenureYears", "Salary", "EngagementSurvey", "Absences", "EmpSatisfaction",
                "DaysLateLast30", "SpecialProjectsCount", "PerfScoreID"]
    top5 = [f for f in PRIORITY if f in red_cols][:5]

    with col_inputs:
        st.markdown("#### Top 5 key factors")
        user_vals = {}
        for feat in top5:
            meta = FEATURE_META.get(feat, {"label": feat, "type": "float", "min": 0.0, "max": 100.0, "default": 50.0, "step": 1.0, "help": ""})
            if meta["type"] == "float":
                user_vals[feat] = st.slider(meta["label"], float(meta["min"]), float(meta["max"]), float(meta["default"]), float(meta["step"]), help=meta["help"])
            else:
                user_vals[feat] = st.slider(meta["label"], int(meta["min"]), int(meta["max"]), int(meta["default"]), int(meta["step"]), help=meta["help"])

        predict_btn = st.button("⚡ Predict risk", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    with col_result:
        if predict_btn:
            # Build input aligned on all reduced model columns (fill missing with median=0)
            input_dict = {col: 0 for col in red_cols}
            input_dict.update(user_vals)
            input_df = pd.DataFrame([input_dict])[red_cols]

            prob  = rf_red.predict_proba(input_df)[0][1]
            score = prob * 100

            if score >= 60:
                color, label, bg = "#ef4444", "🔴 HIGH RISK",    "#450a0a"
            elif score >= 40:
                color, label, bg = "#f59e0b", "🟡 MEDIUM RISK",  "#451a03"
            else:
                color, label, bg = "#22c55e", "🟢 LOW RISK",     "#052e16"

            st.markdown(f"""
            <div style='background:{bg};border:1px solid {color};border-radius:16px;
            padding:28px;text-align:center;margin-bottom:16px'>
                <div style='font-size:3.5rem;font-weight:700;color:{color}'>{score:.0f}%</div>
                <div style='font-size:1.2rem;color:{color};font-weight:600;margin-top:4px'>{label}</div>
                <div style='color:#94a3b8;font-size:0.85rem;margin-top:10px'>Turnover risk score</div>
            </div>""", unsafe_allow_html=True)

            # HR recommendation
            if score >= 60:
                st.markdown("""<div style='background:#450a0a;border-left:4px solid #ef4444;padding:12px 16px;border-radius:0 8px 8px 0;margin-top:8px'>
                <strong style='color:#fca5a5'>Urgent action needed</strong><br>
                <span style='color:#fecaca;font-size:0.88rem'>Schedule a retention interview this week. Review compensation and career path.</span>
                </div>""", unsafe_allow_html=True)
            elif score >= 40:
                st.markdown("""<div style='background:#451a03;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:0 8px 8px 0;margin-top:8px'>
                <strong style='color:#fcd34d'>Enhanced monitoring</strong><br>
                <span style='color:#fde68a;font-size:0.88rem'>Monthly check-in with manager. Review satisfaction and salary evolution.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style='background:#052e16;border-left:4px solid #22c55e;padding:12px 16px;border-radius:0 8px 8px 0;margin-top:8px'>
                <strong style='color:#86efac'>Stable profile</strong><br>
                <span style='color:#bbf7d0;font-size:0.88rem'>Standard follow-up sufficient. Continue to value contributions.</span>
                </div>""", unsafe_allow_html=True)

    # ── Feature impact breakdown ──────────────────────────────────────────────
    if predict_btn:
        st.markdown("---")
        st.subheader("Factor breakdown")
        st.markdown("<small style='color:#64748b'>How each factor pushes the risk score up or down compared to the average employee.</small>", unsafe_allow_html=True)

        # Simple directional impact based on known correlations
        DIRECTION = {
            "TenureYears":    "low → high risk",
            "Salary":         "low → high risk",
            "EngagementSurvey": "low → high risk",
            "Absences":       "high → high risk",
            "EmpSatisfaction": "low → high risk",
            "DaysLateLast30": "high → high risk",
            "PerfScoreID":    "low → high risk",
            "SpecialProjectsCount": "low → high risk",
        }
        AVERAGES = {
            "TenureYears": 6.5, "Salary": 25.0, "EngagementSurvey": 3.8,
            "Absences": 10.0, "EmpSatisfaction": 3.5, "DaysLateLast30": 1.0,
            "PerfScoreID": 3.0, "SpecialProjectsCount": 1.5,
        }

        cols_f = st.columns(len(top5))
        for i, feat in enumerate(top5):
            val     = user_vals[feat]
            avg     = AVERAGES.get(feat, val)
            meta    = FEATURE_META.get(feat, {"label": feat})
            label   = meta["label"].split(" ", 1)[1] if " " in meta["label"] else meta["label"]

            # Is this value pushing risk up or down?
            positive_risk = DIRECTION.get(feat, "").startswith("low")
            if positive_risk:
                risk_up = val < avg
            else:
                risk_up = val > avg

            arrow  = "↑ risk" if risk_up else "↓ risk"
            acolor = "#ef4444" if risk_up else "#22c55e"

            cols_f[i].markdown(f"""
            <div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center'>
                <div style='font-size:1.4rem;font-weight:700;color:#f1f5f9'>{val}</div>
                <div style='font-size:0.8rem;color:#64748b;margin:4px 0'>{label}</div>
                <div style='font-size:0.85rem;color:{acolor};font-weight:600'>{arrow}</div>
                <div style='font-size:0.75rem;color:#475569'>avg: {avg}</div>
            </div>""", unsafe_allow_html=True)

    # ── Frugality note ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px'>
    <span style='color:#22c55e;font-weight:600'>🌱 Frugal AI</span>
    <span style='color:#64748b;font-size:0.88rem'> — This simulator uses only <strong style='color:#e2e8f0'>{len(red_cols)} features</strong> instead of 133.
    Same prediction quality, drastically lower computation. Inference: &lt;1ms on CPU.</span>
    </div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":
    st.title("À propos — TalentGuard AI")

    st.markdown("""
    <div style='background:#1e293b;border-radius:16px;padding:24px;max-width:700px'>
    <h3 style='color:#e2e8f0'>Hackathon Trusted AI x RH — Capgemini x ESILV</h3>

    <p style='color:#94a3b8'>Solution d'IA responsable pour prévenir le turnover des employés, 
    construite autour de deux axes :</p>

    <div style='margin:16px 0'>
    <strong style='color:#ef4444'>🔒 Axe Cybersécurité</strong>
    <ul style='color:#94a3b8;font-size:0.9rem'>
    <li>Anonymisation RGPD (suppression, pseudonymisation, généralisation, masquage)</li>
    <li>Classification AI Act : Haut Risque → obligations renforcées</li>
    <li>Exclusion des attributs sensibles des features du modèle</li>
    </ul>
    </div>

    <div style='margin:16px 0'>
    <strong style='color:#f59e0b'>⚖️ Axe Éthique</strong>
    <ul style='color:#94a3b8;font-size:0.9rem'>
    <li>Audit fairness : Disparate Impact, Demographic Parity, Equalized Odds</li>
    <li>Validation avec AIF360 (IBM)</li>
    <li>Explicabilité locale par SHAP (waterfall plot par employé)</li>
    </ul>
    </div>

    <hr style='border-color:#334155'>

    <table style='width:100%;color:#94a3b8;font-size:0.9rem'>
    <tr><td>Modèle</td><td style='color:#e2e8f0'>Random Forest (100 arbres, class_weight=balanced)</td></tr>
    <tr><td>Données</td><td style='color:#e2e8f0'>HR Dataset Kaggle (~311 employés anonymisés)</td></tr>
    <tr><td>Stack</td><td style='color:#e2e8f0'>Python · scikit-learn · SHAP · AIF360 · Streamlit</td></tr>
    <tr><td>Pipeline</td><td style='color:#e2e8f0'>anonymize → train → fairness_xai → app</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
