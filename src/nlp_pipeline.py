"""
PIPELINE - NLP : Analyse de sentiment des feedbacks RH
=======================================================
Entrée  : data/HRDataset_v14_enriched.csv
Sortie  : data/HR_NLP_scores.csv  (sentiment_score, sentiment_label, risk_score par employé)
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/HRDataset_v14_enriched.csv", encoding="utf-8-sig")
df["ExitInterview_Feedback"]   = df["ExitInterview_Feedback"].fillna("")
df["InternalTransferRequest"]  = df["InternalTransferRequest"].fillna("")
df["Attrition"] = df["EmploymentStatus"].apply(
    lambda x: 1 if x in ("Voluntarily Terminated", "Terminated for Cause") else 0
)
print(f"Dataset chargé : {df.shape[0]} employés")


# ── 2. Lexique sentiment (repris du notebook) ──────────────────────────────────
POSITIVE_WORDS = {
    "positive","appreciated","valued","satisfied","happy","proud",
    "collaborative","supportive","excellent","strong","growth",
    "recognition","flexibility","transparent","engaged","rewarding",
    "motivated","welcoming","enthusiastic","opportunity","constructive",
    "effective","open","balance","healthy","trust","achievement",
}
NEGATIVE_WORDS = {
    "dissatisfied","unhappy","frustrated","micromanaged","overworked",
    "stagnant","ignored","unclear","punitive","conflict","tense",
    "exhausted","unresolved","fear","reprisal","low","broken",
    "hostile","unfair","burnout","disrespected","resign","leaving",
    "quit","toxic","underpaid","undervalued","lack","poor","worse",
}
NEGATIONS = {"not","no","never","neither","nor","without"}

def lexical_sentiment(text, threshold=0.02):
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    if not tokens:
        return {"score": 0.0, "label": "neutral"}
    pos, neg = 0, 0
    for i, token in enumerate(tokens):
        negated = any(tokens[max(0,i-2):i][j] in NEGATIONS for j in range(min(2,i)))
        if token in POSITIVE_WORDS:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif token in NEGATIVE_WORDS:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    score = round((pos - neg) / len(tokens), 4)
    label = "positive" if score > threshold else "negative" if score < -threshold else "neutral"
    return {"score": score, "label": label}

results = df["ExitInterview_Feedback"].apply(lexical_sentiment)
df["sentiment_score"] = results.apply(lambda x: x["score"])
df["sentiment_label"] = results.apply(lambda x: x["label"])


# ── 3. Score de risque combiné (NLP + structuré) ───────────────────────────────
df["nlp_risk_score"] = (
    - df["sentiment_score"] * 10
    + (5 - df["EmpSatisfaction"])
    + (5 - df["EngagementSurvey"])
    + df["Absences"].fillna(0) * 0.1
).round(2)

# Normalisation 0-100 pour comparaison avec le RF
min_s = df["nlp_risk_score"].min()
max_s = df["nlp_risk_score"].max()
df["nlp_risk_pct"] = ((df["nlp_risk_score"] - min_s) / (max_s - min_s) * 100).round(1)


# ── 4. Modèle TF-IDF + Logistic Regression ────────────────────────────────────
mask = df["ExitInterview_Feedback"].str.len() > 20
X_text = df.loc[mask, "ExitInterview_Feedback"]
y      = df.loc[mask, "Attrition"]

nlp_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1,2),
                              stop_words="english", min_df=2, sublinear_tf=True)),
    ("clf",   LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
])
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)
nlp_pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report
print("\n=== Modèle NLP (TF-IDF + LogReg) ===")
print(classification_report(y_test, nlp_pipeline.predict(X_test),
      target_names=["Active","Left"], zero_division=0))


# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────
import os; os.makedirs("models", exist_ok=True)

# Score NLP par employé (pour l'app Streamlit)
# Sauvegarde — on utilise l'index car EmpID n'existe pas dans ce dataset
df[["sentiment_score","sentiment_label","nlp_risk_score","nlp_risk_pct","Attrition"]]\
  .to_csv("data/HR_NLP_scores.csv", index=True)

joblib.dump(nlp_pipeline, "models/nlp_model.joblib")

print("\n✓ data/HR_NLP_scores.csv sauvegardé")
print("✓ models/nlp_model.joblib sauvegardé")