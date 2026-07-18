# 🛡️ FakeRadar — AI-Powered Fake News Detector

## Quick Start (Fully Working — No Setup Needed)

The model is **pre-trained on your dataset**. Just install requirements and run:

```bash
pip install -r requirements.txt
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

---

## Results (Trained on Your Dataset — 44,689 Articles)

| Model               | Accuracy  | F1 Score  | AUC-ROC  |
|---------------------|-----------|-----------|----------|
| **SVM (Linear) ★** | **99.54%** | **99.52%** | **99.98%** |
| Logistic Regression | 98.65%   | 98.58%   | 99.89%  |
| Random Forest       | 98.47%   | 98.38%   | 99.90%  |
| XGBoost             | 98.17%   | 98.07%   | 99.89%  |
| Naive Bayes         | 96.06%   | 95.84%   | 99.15%  |
| Decision Tree       | 95.48%   | 95.26%   | 94.72%  |

---

## Project Structure

```
Fake-News-Detection/
├── dataset/
│   ├── Fake.csv              ← Your dataset (23,481 fake articles)
│   └── True.csv              ← Your dataset (21,417 real articles)
├── model/
│   ├── fake_news_model.pkl   ← Pre-trained SVM model
│   ├── tfidf_vectorizer.pkl  ← Fitted TF-IDF vectorizer
│   └── model_meta.pkl        ← Scores & metadata
├── notebooks/
│   └── Fake_News_Detection_Complete.ipynb
├── static/images/            ← All 6 evaluation charts
├── templates/
│   ├── index.html            ← Main prediction UI
│   └── results.html          ← Dashboard
├── app.py                    ← Flask web app
├── train.py                  ← Full re-training pipeline
├── preprocessing.py          ← NLP pipeline
└── requirements.txt
```

---

## Re-train From Scratch

```bash
python train.py
```

## REST API

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your article text here..."}'
```

Response:
```json
{"label": 1, "label_text": "Real News", "confidence_real": 94.5, "confidence_fake": 5.5, "latency_ms": 1.2}
```
