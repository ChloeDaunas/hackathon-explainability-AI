# 🛡️ TalentGuard AI — Hackathon Trusted AI x RH

> **Capgemini x ESILV** · 16-17 mars 2025  
> Solution d'IA responsable pour prévenir le turnover des employés

---

## 🎯 Objectif

Une entreprise imaginaire fait face à un taux de démissions élevé. **TalentGuard AI** est une solution qui aide la direction RH à :

- **Identifier** les employés à risque de démission
- **Comprendre** les facteurs explicatifs (pas de boîte noire)
- **Agir** de façon éthique, transparente et conforme au RGPD

Le projet couvre **deux axes Trusted AI** :

| Axe | Ce qu'on fait |
|-----|---------------|
| 🔒 **Cybersécurité** | Anonymisation RGPD, classification EU AI Act, exclusion des attributs sensibles |
| ⚖️ **Éthique** | Audit fairness (genre, ethnie), explicabilité SHAP, conformité AI Act |

---

## 👤 Personas & Use Case

**Use Case principal** : *Analyse et prévention du risque de démission*

**Persona 1 — Responsable RH**
> *"J'ai besoin de savoir quels employés risquent de partir, pourquoi, et comment agir — sans discriminer."*

**Persona 2 — Direction / DSI**
> *"Je veux une solution IA fiable, conforme RGPD et explicable pour mes équipes."*

---

## 🗂️ Architecture du projet

```
talentguard-ai/
├── app.py                          # Application Streamlit (démo)
├── requirements.txt                # Dépendances Python
├── README.md
│
├── data/
│   ├── HRDataset_v14.csv           # Dataset original (Kaggle - Rich Huebner)
│   ├── HR_Dataset_Anonymized.csv   # Généré par anonymize_data.py
│   ├── fairness_audit_results.csv  # Généré par fairness_xai.py
│   ├── shap_feature_importance.csv # Généré par fairness_xai.py
│   ├── shap_global.png             # Généré par fairness_xai.py
│   └── shap_local.png              # Généré par fairness_xai.py
│
├── models/
│   ├── rf_model.joblib             # Modèle entraîné (généré par train_model.py)
│   └── model_feature_columns.json # Colonnes du modèle
│
└── src/
    ├── anonymize_data.py           # Étape 1 : anonymisation RGPD
    ├── train_model.py              # Étape 2 : entraînement Random Forest
    └── fairness_xai.py             # Étape 3 : audit fairness + SHAP
```

---

## ⚙️ Installation

**Prérequis** : Python 3.10+

```bash
# 1. Cloner le repo
git clone <URL_DU_REPO>
cd talentguard-ai

# 2. Créer et activer le venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancer le pipeline

Exécuter les scripts dans l'ordre :

```bash
# Étape 1 — Anonymisation des données (Axe Cybersécurité)
python src/anonymize_data.py

# Étape 2 — Entraînement du modèle ML
python src/train_model.py

# Étape 3 — Audit Fairness + SHAP (Axe Éthique)
python src/fairness_xai.py

# Étape 4 — Lancer l'application
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`

---

## 📊 Données

**Source** : [Human Resources Dataset — Kaggle (Rich Huebner & Carla Patalano)](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)

- ~311 employés (après anonymisation)
- Variable cible : `Termd` (0 = actif, 1 = a quitté)
- Taux de turnover : ~33%
- Attributs sensibles présents : `Sex`, `RaceDesc`, `GenderID`

**Data Card — Transformations appliquées :**

| Technique | Colonnes concernées | Justification |
|-----------|--------------------|----|
| Suppression | `Employee_Name`, `ManagerName` | Identifiants directs |
| Pseudonymisation | `EmpID` | Hash SHA-256 tronqué, non réversible |
| Généralisation | `DOB` → `AgeGroup` | Info utile sans date exacte |
| Masquage | `Zip` → 2 premiers chiffres | Réduction de précision géographique |
| Exclusion modèle | `Sex`, `RaceDesc`, `GenderID` | Conservés uniquement pour l'audit fairness |

---

## 🤖 Model Card

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | Random Forest Classifier |
| Librairie | scikit-learn 1.5.0 |
| `n_estimators` | 100 |
| `class_weight` | balanced (classes déséquilibrées) |
| Split | 80% train / 20% test (stratifié) |
| Features | 132 colonnes après one-hot encoding |
| **Precision (Left)** | **0.72** |
| **Recall (Left)** | **0.62** |
| **F1-score (Left)** | **0.67** |
| **Accuracy** | **0.79** |

**Top features (SHAP)** : `TenureYears`, `Salary`, `EngagementSurvey`, `Absences`, `EmpSatisfaction`

> Le recall est la métrique prioritaire : manquer un vrai départ est plus coûteux qu'une fausse alerte.

---

## ⚖️ Audit Fairness

Métriques calculées avec les méthodes AIF360 (IBM) sur le test set (63 employés) :

| Attribut | Disparate Impact | Dem. Parity Diff | Eq. Odds Diff | Verdict |
|----------|-----------------|-----------------|----------------|---------|
| Genre (M vs F) | 1.066 | +0.018 | 0.155 | ✅ DI OK · ⚠️ EOD à surveiller |
| Ethnie (White vs autres) | 0.826 | -0.054 | 0.155 | ✅ DI OK · ⚠️ EOD à surveiller |

**Règles d'interprétation :**
- Disparate Impact : acceptable entre **0.8 et 1.25** (règle des 80%)
- Demographic Parity Diff : idéal = **0**, seuil d'alerte = **±0.1**
- Equalized Odds Diff : idéal = **0**, seuil d'alerte = **0.1**

---

## 🇪🇺 Classification EU AI Act

Ce système est classifié **HAUT RISQUE** (Annexe III — systèmes RH et gestion de l'emploi).

**Obligations appliquées dans ce projet :**
- ✅ Supervision humaine obligatoire (recommandations, pas de décision automatique)
- ✅ Traçabilité des prédictions (scores sauvegardés)
- ✅ Audit de fairness documenté (`fairness_audit_results.csv`)
- ✅ Droit à l'explication (waterfall SHAP par employé dans l'app)

---

## 🖥️ Application — Pages

| Page | Contenu |
|------|---------|
| 📊 **Dashboard** | KPIs globaux, distribution des scores de risque, top 10 employés à risque |
| 🔍 **Analyse individuelle** | Score par employé, explication SHAP, recommandation RH automatique |
| ⚖️ **Audit Fairness** | Métriques par groupe (genre/ethnie), graphiques, classification AI Act |
| ℹ️ **À propos** | Stack technique, résumé du projet |

---

## 🛠️ Stack technique

```
Python 3.10
├── pandas          → manipulation des données
├── scikit-learn    → modèle ML (Random Forest)
├── shap            → explicabilité (SHAP values, waterfall plots)
├── aif360          → audit fairness (IBM)
├── fairlearn       → métriques complémentaires
├── streamlit       → application web
└── matplotlib      → visualisations
```

---

## 👥 Équipe

Hackathon Trusted AI x RH — ESILV · Capgemini · Mars 2025
