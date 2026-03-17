# 📧 Email Spam Classifier — End-to-End ML Project

Classify emails and SMS messages as **spam** or **ham** using a full NLP pipeline and multiple classifiers.

---

## 📁 Project Structure

```
email_classification/
│
├── data/raw/
├── models/                ← Saved .pkl artifacts
├── logs/                  ← pipeline.log + model_comparison.csv
│
├── src/
│   ├── text_preprocessing.py  ← NLP pipeline + TF-IDF
│   ├── data_loader.py         ← Load, clean, encode, split
│   ├── model_training.py      ← Train, save, load models
│   ├── model_evaluation.py    ← Metrics & comparison
│   ├── predict.py             ← Inference helpers
│   └── logger.py              ← Logging setup
│
├── config.py              ← All settings in one place
├── train.py               ← Training entrypoint
├── predict_cli.py         ← CLI classifier
├── app.py                 ← Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) → `data/raw/spam.csv`

---

## 🚀 Train

```bash
python train.py
```

Pipeline: **load → clean → encode → split → preprocess text → TF-IDF → train 4 models → evaluate → save best**

```
═════════════════════════════════════════════════════════════════
  MODEL COMPARISON  (sorted by F1)
═════════════════════════════════════════════════════════════════
              Model  Accuracy  Precision  Recall      F1
  LogisticRegression    0.9847     0.9849  0.9847  0.9847
                 SVM    0.9830     0.9833  0.9830  0.9829
          NaiveBayes    0.9784     0.9787  0.9784  0.9783
        RandomForest    0.9713     0.9718  0.9713  0.9710
═════════════════════════════════════════════════════════════════
```

---

## 🔮 Predict

```bash
# Single email
python predict_cli.py --text "WINNER! Claim your free prize now!"

# From file (one email per line)
python predict_cli.py --file emails.txt

# Interactive
python predict_cli.py
```

---

## 🎨 Streamlit Dashboard

```bash
streamlit run app.py
```

| Page | Features |
|------|----------|
| 🔍 **Classifier** | Paste email → instant verdict + confidence gauge + batch mode |
| 📊 **Dataset Explorer** | Pie chart, length distributions, top spam words |
| 🤖 **Model Insights** | F1 bar chart, radar chart, pipeline steps walkthrough |

---

## 🧪 Tests

```bash
python tests/test_pipeline.py
```
9 smoke tests covering the full pipeline. Also run automatically via GitHub Actions CI.

---

## 🔧 Configuration

Edit `config.py` to change:
- Number of TF-IDF features (`TFIDF_MAX_FEATURES`)
- Enable/disable stemming or lemmatization
- Add/remove models from `MODEL_CONFIGS`
- Paths and split ratios

---

## 🧩 NLP Pipeline

```
Raw text
  │  lowercase
  │  tokenize (NLTK punkt)
  │  remove punctuation & digits
  │  remove stopwords
  │  stem (PorterStemmer)
  │  lemmatize (WordNetLemmatizer)
  │  join tokens → string
  └→ TF-IDF (5,000 features) → Model → spam / ham
```
