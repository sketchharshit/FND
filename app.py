"""
app.py  — Flask web application for AI-Powered Fake News Detector
Run: python app.py   →  http://127.0.0.1:5000
"""
import os, pickle, time, numpy as np
from flask import Flask, request, render_template, jsonify
from preprocessing import preprocess_text

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

def _load(fn):
    p = os.path.join(MODEL_DIR, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}\nRun train.py first.")
    with open(p,"rb") as f: return pickle.load(f)

MODEL      = _load("fake_news_model.pkl")
VECTORIZER = _load("tfidf_vectorizer.pkl")
META       = _load("model_meta.pkl")

# Warm-up: load NLTK corpora now, not on first request
_w = preprocess_text("warm up the nlp pipeline on startup")
VECTORIZER.transform([_w])
print(f"✓ {META['model_name']} loaded — Acc={META['accuracy']}%  F1={META['f1_score']}%")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

def predict_article(raw: str) -> dict:
    t0 = time.perf_counter()
    cleaned  = preprocess_text(raw)
    features = VECTORIZER.transform([cleaned])
    label    = int(MODEL.predict(features)[0])
    if META["is_proba_clf"]:
        p = MODEL.predict_proba(features)[0]; cf, cr = float(p[0]), float(p[1])
    elif hasattr(MODEL, "decision_function"):
        raw_s = float(MODEL.decision_function(features)[0])
        cr = float(1/(1+np.exp(-raw_s))); cf = 1.0-cr
    else:
        cr = float(label); cf = 1.0-cr
    return {"label":label, "label_text":"Real News" if label==1 else "Fake News",
            "confidence_fake":round(cf*100,2), "confidence_real":round(cr*100,2),
            "word_count":len(raw.split()), "processed_word_count":len(cleaned.split()),
            "latency_ms":round((time.perf_counter()-t0)*1000,1)}

@app.route("/")
def index(): return render_template("index.html", meta=META)

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        raw = (request.get_json(silent=True) or {}).get("text","").strip()
    else:
        raw = (request.form.get("article_text") or "").strip()
    if not raw:             return jsonify({"error":"No text provided."}), 400
    if len(raw) < 50:       return jsonify({"error":"Article too short (need ≥ 50 characters)."}), 400
    return jsonify(predict_article(raw))

@app.route("/results")
def results():
    rdf = META.get("results_df")
    rows = []
    if rdf is not None:
        for _,row in rdf.iterrows():
            rows.append({"model":row["Model"],"accuracy":row["Accuracy"],
                "precision":row["Precision"],"recall":row["Recall"],
                "f1_score":row["F1 Score"],"auc_roc":row["AUC-ROC"],
                "train_time":row["Train Time"],"is_best":row["Model"]==META["model_name"]})
    return render_template("results.html", meta=META, table_rows=rows)

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","model":META["model_name"],
                    "accuracy":META["accuracy"],"f1":META["f1_score"]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
