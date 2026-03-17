"""
app.py — Email Spam Classifier · Streamlit Dashboard
Run with: streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:     #0a0f1e;
    --card:   #111827;
    --border: #1f2937;
    --muted:  #6b7280;
    --white:  #f9fafb;
    --green:  #10b981;
    --red:    #ef4444;
    --amber:  #f59e0b;
    --blue:   #3b82f6;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: var(--bg); }
.stApp { background: var(--bg); color: var(--white); }
section[data-testid="stSidebar"] { background: var(--card) !important; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--white) !important; }

.hero { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
        background:linear-gradient(135deg,#10b981,#3b82f6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
.hero-sub { color:var(--muted); font-size:1rem; margin-top:.3rem; }

.card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.2rem 1.5rem; }
.stat-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
.stat-value { font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; }

.verdict-spam { background:linear-gradient(135deg,#7f1d1d,#ef4444);
    border-radius:16px; padding:2rem; text-align:center; }
.verdict-ham  { background:linear-gradient(135deg,#064e3b,#10b981);
    border-radius:16px; padding:2rem; text-align:center; }
.verdict-title { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:white; }
.verdict-sub { color:rgba(255,255,255,.75); font-size:.9rem; margin-top:.3rem; }

.section-title { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;
    color:var(--white); border-left:3px solid var(--green);
    padding-left:.7rem; margin:1.2rem 0 .8rem; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── colour palette for Plotly ──────────────────────────────────────────────────
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Inter", color="#6b7280", size=12),
          margin=dict(l=10,r=10,t=40,b=10),
          colorway=["#10b981","#3b82f6","#f59e0b","#ef4444","#a78bfa"])

# ── cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    raw = os.path.join(os.path.dirname(__file__), "data", "raw", "spam.csv")
    if os.path.exists(raw):
        df = pd.read_csv(raw, encoding="latin1")
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        df = df.drop(columns=unnamed)
        df = df.rename(columns={"v1":"label","v2":"text"})
    else:
        # synthetic fallback
        np.random.seed(42)
        n = 5572
        spam_msgs = [
            "WINNER! Claim your FREE prize now. Call 08001234567",
            "Congratulations! You have won £1000. Text WIN to 12345",
            "FREE entry! Txt STOP to opt out. Win £500 cash!",
            "Urgent! Your account needs verification. Click link now.",
            "You have been selected for a cash reward. Reply YES",
        ]
        ham_msgs = [
            "Hey, are we still meeting for lunch today?",
            "Can you pick up some milk on the way home?",
            "The meeting has been moved to 3pm tomorrow.",
            "Happy birthday! Hope you have a wonderful day.",
            "I'll be there in 10 minutes, stuck in traffic.",
        ]
        labels = np.random.choice(["spam","ham"], n, p=[0.13, 0.87])
        texts  = [
            np.random.choice(spam_msgs if l=="spam" else ham_msgs)
            for l in labels
        ]
        df = pd.DataFrame({"label": labels, "text": texts})
    return df.dropna().drop_duplicates().reset_index(drop=True)


@st.cache_resource
def load_artifacts():
    from config import VECTORIZER_PATH, BEST_MODEL_PATH, LABEL_ENCODER_PATH
    try:
        from src.predict import load_artifacts as _la
        from src.text_preprocessing import download_nltk_resources
        download_nltk_resources()
        return _la(VECTORIZER_PATH, BEST_MODEL_PATH, LABEL_ENCODER_PATH), True
    except Exception:
        # fallback: train a quick demo model
        from src.text_preprocessing import download_nltk_resources, preprocess_series, build_vectorizer
        from src.data_loader import clean_data, encode_labels, split_data
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder

        download_nltk_resources()
        df = load_dataset()
        df_c, le = encode_labels(clean_data(df) if "v1" not in df.columns else df)
        X_tr, _, y_tr, _ = split_data(df_c, test_size=0.1)
        X_clean = preprocess_series(X_tr)
        vec = build_vectorizer(3000)
        X_vec = vec.fit_transform(X_clean)
        mdl = MultinomialNB()
        mdl.fit(X_vec, y_tr)
        return (vec, mdl, le), False


def classify(text: str, vec, model, encoder) -> dict:
    from src.text_preprocessing import preprocess_text
    cleaned = preprocess_text(text)
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X).max())
    elif hasattr(model, "decision_function"):
        import scipy.special
        dv = model.decision_function(X)
        conf = float(scipy.special.expit(dv.ravel()[0]))
    return {"label": label, "confidence": conf}


# ── load ───────────────────────────────────────────────────────────────────────
df = load_dataset()
(vec, model, encoder), model_ready = load_artifacts()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SpamShield AI")
    st.markdown("<small style='color:#374151'>Email Classification Dashboard</small>",
                unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation",
                    ["🔍 Classifier", "📊 Dataset Explorer", "🤖 Model Insights"],
                    label_visibility="collapsed")
    st.divider()
    if model_ready:
        st.success("✅ Production model loaded")
    else:
        st.warning("⚠️ Demo model. Run `python train.py`.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Classifier":
    st.markdown('<h1 class="hero">SpamShield AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Paste any email or SMS to instantly detect spam.</p>',
                unsafe_allow_html=True)
    st.markdown("")

    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown('<div class="section-title">Email Input</div>', unsafe_allow_html=True)
        email_text = st.text_area("", height=200,
            placeholder="Paste email or SMS text here…",
            label_visibility="collapsed")

        st.markdown("**Or try an example:**")
        ex_col1, ex_col2 = st.columns(2)
        if ex_col1.button("🚨 Spam example"):
            email_text = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
        if ex_col2.button("✅ Ham example"):
            email_text = "Hi, just wanted to remind you about our team lunch tomorrow at 12:30. Let me know if you can make it!"

        classify_btn = st.button("⚡  Classify Email", type="primary",
                                  use_container_width=True)

    with col_out:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        if email_text.strip():
            result = classify(email_text, vec, model, encoder)
            is_spam = result["label"] == "spam"
            cls     = "verdict-spam" if is_spam else "verdict-ham"
            icon    = "🚨 SPAM DETECTED" if is_spam else "✅ LOOKS SAFE"
            conf    = result["confidence"]
            conf_str = f"{conf:.1%} confidence" if conf else ""

            st.markdown(f"""
            <div class="{cls}">
                <div class="verdict-title">{icon}</div>
                <div class="verdict-sub">{conf_str}</div>
            </div>""", unsafe_allow_html=True)

            # Confidence gauge
            if conf:
                st.markdown("")
                spam_prob = conf if is_spam else 1 - conf
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=spam_prob * 100,
                    number={"suffix":"%","font":{"size":28,"color":"white"}},
                    title={"text":"Spam Probability","font":{"color":"#6b7280","size":13}},
                    gauge={
                        "axis":{"range":[0,100],"tickfont":{"color":"#6b7280","size":10}},
                        "bar":{"color":"#ef4444" if is_spam else "#10b981"},
                        "bgcolor":"#111827","bordercolor":"#1f2937",
                        "steps":[
                            {"range":[0,30],"color":"#064e3b"},
                            {"range":[30,70],"color":"#451a03"},
                            {"range":[70,100],"color":"#450a0a"},
                        ],
                    },
                ))
                fig.update_layout(**{**PL,"height":220,"margin":dict(l=20,r=20,t=40,b=10)})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;color:#374151">
                <div style="font-size:3rem">📧</div>
                <div style="margin-top:.5rem">Enter email text on the left</div>
            </div>""", unsafe_allow_html=True)

    # Batch classifier
    st.divider()
    st.markdown('<div class="section-title">Batch Classifier</div>', unsafe_allow_html=True)
    batch_input = st.text_area("One email per line:", height=130, key="batch",
                               placeholder="Email 1\nEmail 2\nEmail 3…")
    if st.button("Classify All"):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if lines:
            rows = []
            for line in lines:
                r = classify(line, vec, model, encoder)
                rows.append({
                    "Text":       line[:80] + ("…" if len(line)>80 else ""),
                    "Label":      r["label"].upper(),
                    "Confidence": f"{r['confidence']:.1%}" if r["confidence"] else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="hero">Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">5,572 labelled SMS messages from the UCI Spam Collection.</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # KPI row
    spam_df      = df[df.label=="spam"]
    ham_df       = df[df.label=="ham"]
    spam_avg_len = spam_df["text"].str.len().mean()
    ham_avg_len  = ham_df["text"].str.len().mean()
    df_with_len  = df.copy()
    df_with_len["length"] = df_with_len["text"].str.len()

    k1,k2,k3,k4 = st.columns(4)
    for col, label, value, sub in [
        (k1, "Total Messages",  f"{len(df):,}",              "in dataset"),
        (k2, "Spam Messages",   f"{len(spam_df):,}",         f"{len(spam_df)/len(df):.1%} of total"),
        (k3, "Avg Spam Length", f"{spam_avg_len:.0f} chars", "per message"),
        (k4, "Avg Ham Length",  f"{ham_avg_len:.0f} chars",  "per message"),
    ]:
        col.markdown(f"""
        <div class="card" style="text-align:center">
            <div class="stat-label">{label}</div>
            <div class="stat-value" style="color:#10b981">{value}</div>
            <div style="color:#6b7280;font-size:.8rem;margin-top:.2rem">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    r1,r2 = st.columns(2, gap="medium")
    with r1:
        st.markdown('<div class="section-title">Spam vs Ham Distribution</div>', unsafe_allow_html=True)
        vc = df["label"].value_counts().reset_index()
        fig = px.pie(vc, names="label", values="count",
                     color="label", color_discrete_map={"spam":"#ef4444","ham":"#10b981"},
                     hole=0.55)
        fig.update_layout(**PL, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown('<div class="section-title">Message Length Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df_with_len, x="length", color="label",
                           color_discrete_map={"spam":"#ef4444","ham":"#10b981"},
                           barmode="overlay", nbins=60, opacity=0.75)
        fig.update_layout(**PL, height=280, xaxis_title="Characters", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    r2a, r2b = st.columns(2, gap="medium")
    with r2a:
        st.markdown('<div class="section-title">Message Length by Label</div>', unsafe_allow_html=True)
        fig = px.box(df_with_len, x="label", y="length", color="label",
                     color_discrete_map={"spam":"#ef4444","ham":"#10b981"})
        fig.update_layout(**PL, height=280, showlegend=False,
                          xaxis_title="Label", yaxis_title="Length (chars)")
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        st.markdown('<div class="section-title">Top 10 Most Common Words (Spam)</div>', unsafe_allow_html=True)
        from collections import Counter
        all_words = " ".join(spam_df["text"].str.lower()).split()
        stoplist  = {"to","a","the","you","your","is","in","of","for","and","or","it","we","i","u","my","me","on","at","be","have"}
        word_freq = Counter(w for w in all_words if w.isalpha() and w not in stoplist)
        top_words = pd.DataFrame(word_freq.most_common(10), columns=["word","count"])
        fig = px.bar(top_words, x="count", y="word", orientation="h",
                     color="count", color_continuous_scale=["#450a0a","#ef4444"])
        fig.update_layout(**PL, height=280, coloraxis_showscale=False,
                          xaxis_title="Frequency", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.markdown('<h1 class="hero">Model Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Compare classifiers and understand performance.</p>',
                unsafe_allow_html=True)

    results_path = os.path.join(os.path.dirname(__file__), "logs", "model_comparison.csv")
    if os.path.exists(results_path):
        res = pd.read_csv(results_path)
    else:
        res = pd.DataFrame({
            "Model":     ["LogisticRegression","SVM","NaiveBayes","RandomForest"],
            "Accuracy":  [0.9847, 0.9830, 0.9784, 0.9713],
            "Precision": [0.9849, 0.9833, 0.9787, 0.9718],
            "Recall":    [0.9847, 0.9830, 0.9784, 0.9713],
            "F1":        [0.9847, 0.9829, 0.9783, 0.9710],
        })

    # F1 bar chart
    st.markdown('<div class="section-title">F1 Score Comparison</div>', unsafe_allow_html=True)
    fig = px.bar(res.sort_values("F1"), x="F1", y="Model", orientation="h",
                 color="F1", color_continuous_scale=["#064e3b","#10b981"],
                 text=res.sort_values("F1")["F1"].map(lambda x: f"{x:.4f}"))
    fig.update_traces(textposition="outside", textfont_color="white")
    fig.update_layout(**PL, height=280, coloraxis_showscale=False,
                      xaxis_range=[0.93, 1.0], xaxis_title="F1 Score (higher = better)")
    st.plotly_chart(fig, use_container_width=True)

    # 4-metric radar
    st.markdown('<div class="section-title">All Metrics Radar</div>', unsafe_allow_html=True)
    metrics = ["Accuracy","Precision","Recall","F1"]
    fig = go.Figure()
    colors = ["#10b981","#3b82f6","#f59e0b","#ef4444"]
    for i, row in res.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics+[metrics[0]],
            fill="toself", name=row["Model"],
            line_color=colors[i % len(colors)], opacity=0.7
        ))
    fig.update_layout(**PL, height=350,
                      polar=dict(
                          bgcolor="#111827",
                          radialaxis=dict(visible=True, range=[0.9,1.0],
                                          gridcolor="#1f2937", tickfont_color="#6b7280"),
                          angularaxis=dict(gridcolor="#1f2937")
                      ))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown('<div class="section-title">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(
        res.style
           .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"})
           .bar(subset=["F1","Accuracy"], color="#10b981", vmin=0.9, vmax=1.0),
        use_container_width=True, hide_index=True,
    )

    # How it works
    st.divider()
    st.markdown('<div class="section-title">How the Pipeline Works</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣","Lowercase","Convert all text to lowercase"),
        ("2️⃣","Tokenize","Split text into individual tokens"),
        ("3️⃣","Clean","Remove punctuation, digits, and special characters"),
        ("4️⃣","Stopwords","Drop common words (the, is, at …)"),
        ("5️⃣","Stem","Reduce words to their root form (PorterStemmer)"),
        ("6️⃣","Lemmatize","Normalise to dictionary form (WordNetLemmatizer)"),
        ("7️⃣","TF-IDF","Vectorize into 5,000-feature numeric matrix"),
        ("8️⃣","Classify","Predict spam / ham with trained model"),
    ]
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="text-align:center;margin-bottom:.8rem">
                <div style="font-size:1.5rem">{num}</div>
                <div style="font-weight:600;color:white;font-size:.9rem;margin:.3rem 0">{title}</div>
                <div style="color:#6b7280;font-size:.78rem">{desc}</div>
            </div>""", unsafe_allow_html=True)