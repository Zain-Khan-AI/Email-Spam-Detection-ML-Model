import streamlit as st
import string
import nltk
import pickle
import html
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="MailGuard | Spam Intelligence",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- MODEL / NLP --------------------
@st.cache_resource
def load_assets():
    base_dir = Path(__file__).parent
    with open(base_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(base_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

@st.cache_resource
def get_nlp_assets():
    # Downloads the required NLTK resources if they are not installed.
    for resource, package in [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
    return set(stopwords.words("english")), PorterStemmer()

def transform_text(text, stop_words, stemmer):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [
        token for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]
    return " ".join(stemmer.stem(token) for token in tokens)

try:
    tfidf, model = load_assets()
    stop_words, ps = get_nlp_assets()
except Exception as exc:
    st.error(
        "The model could not be loaded. Make sure `model.pkl` and "
        "`vectorizer.pkl` are in the same folder as this Streamlit app, "
        "and that the required NLTK data is available."
    )
    st.exception(exc)
    st.stop()

# -------------------- PROFESSIONAL UI --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.social-links {
    display: flex;
    align-items: center;
    gap: 10px;
}

.social-link {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(157,178,211,.2);
    border-radius: 12px;
    color: #DCE6F5;
    background: rgba(255,255,255,.03);
    text-decoration: none !important;
    transition: all .2s ease;
}

.social-link:hover {
    color: #63E6D3;
    border-color: #63E6D3;
    background: rgba(99,230,211,.08);
}

:root {
    --bg: #080D18;
    --panel: #101827;
    --panel2: #121D2F;
    --line: rgba(157,178,211,.14);
    --muted: #8D9BB3;
    --text: #F3F6FC;
    --cyan: #63E6D3;
    --blue: #8AA7FF;
}
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(61,91,166,.19), transparent 32%),
      radial-gradient(ellipse at 92% 10%, rgba(43,183,164,.09), transparent 28%),
      var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1420px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3,p,div,span,label { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.brand-row {
    display:flex; align-items:center; justify-content:space-between;
    gap:18px; padding: 3px 0 25px; border-bottom:1px solid var(--line);
    margin-bottom:30px;
}
.brand {display:flex; align-items:center; gap:12px;}
.brand-icon {
    width:44px;height:44px;border-radius:14px;display:flex;align-items:center;
    justify-content:center;background:linear-gradient(135deg,#6CE7D5,#76A2FF);
    box-shadow:0 8px 30px rgba(99,230,211,.16);
}
.brand-name {font-family:'Manrope',sans-serif;font-size:21px;font-weight:800;letter-spacing:-.6px;color:#F6F8FC;}
.brand-sub {font-size:11px;color:var(--muted);letter-spacing:1.8px;text-transform:uppercase;margin-top:2px;}
.top-status {border:1px solid rgba(99,230,211,.25);background:rgba(99,230,211,.07);color:#8BF0DE;border-radius:30px;padding:8px 13px;font-size:12px;font-weight:600;}
.hero-kicker {color:var(--cyan);text-transform:uppercase;letter-spacing:2.5px;font-size:11px;font-weight:700;margin-bottom:12px;}
.hero-title {font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-1.8px;line-height:1.08;font-size:clamp(32px,4vw,52px);margin:0;color:#F5F7FC;}
.hero-title span {color:var(--cyan);}
.hero-copy {color:#9CAAC0;font-size:15px;line-height:1.75;max-width:640px;margin-top:15px;}
.section-label {color:#9BAAC1;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:0 0 14px;}
.metric-card {
    background:linear-gradient(145deg,rgba(20,31,49,.96),rgba(13,21,35,.96));
    border:1px solid var(--line);border-radius:18px;padding:21px 22px;min-height:125px;
}
.metric-top {display:flex;justify-content:space-between;align-items:center;gap:10px;color:#A0AEC2;font-size:12px;}
.metric-icon {width:32px;height:32px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:rgba(138,167,255,.11);color:var(--blue);}
.metric-value {font-family:'Manrope',sans-serif;font-size:31px;font-weight:800;letter-spacing:-1px;color:#F4F7FC;margin-top:10px;}
.metric-foot {font-size:11px;color:#7E8DA5;margin-top:3px;}
.panel {
    background:linear-gradient(145deg,rgba(17,27,43,.98),rgba(12,20,33,.98));
    border:1px solid var(--line);border-radius:22px;padding:25px;
}
.panel-title {font-family:'Manrope',sans-serif;font-size:19px;font-weight:800;color:#F4F7FC;letter-spacing:-.4px;}
.panel-desc {font-size:12px;color:#8D9BB3;line-height:1.6;margin-top:5px;margin-bottom:21px;}
.input-hint {font-size:12px;color:#8D9BB3;margin:0 0 8px;}
.stTextArea textarea {
    background:#0A1220!important;color:#EDF3FC!important;
    border:1px solid #29364B!important;border-radius:14px!important;
    padding:16px!important;font-size:14px!important;line-height:1.7!important;
}
.stTextArea textarea:focus {border-color:#63E6D3!important;box-shadow:0 0 0 1px #63E6D3!important;}
.stButton > button {
    width:100%;min-height:48px;border:0;border-radius:13px;
    background:linear-gradient(100deg,#6CE7D5,#8AA7FF);
    color:#07121D;font-weight:800;font-size:14px;
    transition:all .2s ease;
}
.stButton > button:hover {filter:brightness(1.07);transform:translateY(-1px);color:#07121D;}
.result-card {border-radius:18px;padding:22px;border:1px solid var(--line);background:#0B1422;margin-top:12px;}
.result-spam {border-color:rgba(255,105,125,.35);background:linear-gradient(140deg,rgba(109,31,50,.25),rgba(15,22,36,.98));}
.result-safe {border-color:rgba(99,230,211,.3);background:linear-gradient(140deg,rgba(28,104,92,.20),rgba(15,22,36,.98));}
.result-label {font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#91A0B7;font-weight:700;}
.result-main {font-family:'Manrope',sans-serif;font-size:28px;font-weight:800;margin-top:9px;}
.result-note {font-size:13px;line-height:1.65;color:#AAB6C8;margin-top:7px;}
.result-badge {display:inline-block;border-radius:30px;padding:6px 10px;font-size:11px;font-weight:800;letter-spacing:.5px;margin-top:14px;}
.badge-spam {background:rgba(255,105,125,.13);color:#FF8FA0;}
.badge-safe {background:rgba(99,230,211,.12);color:#75E8D6;}
.info-row {display:flex;align-items:flex-start;gap:12px;margin:17px 0;}
.info-dot {width:8px;height:8px;border-radius:50%;background:var(--cyan);margin-top:6px;flex-shrink:0;}
.info-heading {font-size:13px;font-weight:700;color:#E6ECF6;}
.info-copy {font-size:12px;color:#8D9BB3;line-height:1.6;margin-top:4px;}
.footer {border-top:1px solid var(--line);margin-top:30px;padding-top:18px;color:#687890;font-size:11px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;}
div[data-testid="stHorizontalBlock"] {gap:16px;}
@media (max-width: 700px) {
  .block-container {padding:1rem 1rem 2rem;}
  .panel {padding:18px;}
  .brand-sub {letter-spacing:1px;}
  .top-status {font-size:10px;padding:7px 9px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-row">
  <div class="brand">
    <div class="brand-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-11Z" stroke="#07121D" stroke-width="1.8"/>
        <path d="m5 7 7 5.5L19 7" stroke="#07121D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="m15.5 16 1.5 1.5 3-3.5" stroke="#07121D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div><div class="brand-name">MailGuard<span style="color:#63E6D3">.</span></div>
    <div class="brand-sub">Email security intelligence</div></div>
  </div>
  
<div style="display:flex; align-items:center; gap:10px;">
  <div class="social-links">
    <a class="social-link"
       href="https://www.linkedin.com/in/m-zainkhan/"
       target="_blank"
       rel="noopener noreferrer"
       title="Visit my LinkedIn profile"
       aria-label="LinkedIn profile">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20.45 2H3.55C2.69 2 2 2.68 2 3.52v16.96c0 .84.69 1.52 1.55 1.52h16.9c.86 0 1.55-.68 1.55-1.52V3.52C22 2.68 21.31 2 20.45 2ZM8 19H5V9h3v10ZM6.5 7.7a1.75 1.75 0 1 1 0-3.5 1.75 1.75 0 0 1 0 3.5ZM19 19h-3v-4.9c0-1.17-.02-2.68-1.63-2.68-1.63 0-1.87 1.27-1.87 2.59V19h-3V9h2.88v1.37h.04c.4-.76 1.38-1.56 2.84-1.56C18.3 8.81 19 10.74 19 13.25V19Z"/>
      </svg>
    </a>


<div style="display:flex; align-items:center; gap:10px;">
  <a href="https://github.com/Zain-Khan-AI"
     target="_blank"
     rel="noopener noreferrer"
     style="display:flex;align-items:center;justify-content:center;width:40px;height:40px;border:1px solid rgba(157,178,211,.2);border-radius:12px;color:#DCE6F5;background:rgba(255,255,255,.03);text-decoration:none;"
     title="GitHub Profile">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.76-.24.76-.54v-2.1c-3.1.67-3.76-1.32-3.76-1.32-.5-1.29-1.24-1.63-1.24-1.63-1.01-.69.08-.68.08-.68 1.12.08 1.71 1.15 1.71 1.15 1 1.71 2.62 1.22 3.26.93.1-.72.39-1.22.71-1.5-2.48-.28-5.09-1.24-5.09-5.52 0-1.22.44-2.22 1.15-3-.12-.28-.5-1.42.11-2.96 0 0 .94-.3 3.05 1.15a10.6 10.6 0 0 1 5.55 0c2.11-1.45 3.05-1.15 3.05-1.15.61 1.54.23 2.68.11 2.96.71.78 1.15 1.78 1.15 3 0 4.29-2.61 5.24-5.1 5.51.4.35.76 1.03.76 2.08v3.08c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z"/>
    </svg>
  </a>
  <div class="top-status">● &nbsp; CLASSIFIER READY</div>
</div>

  </div>

</div>

</div>

<div class="hero-kicker">Machine learning · Text classification</div>
<div class="hero-title">Your inbox deserves<br/><span>smarter protection.</span></div>
<div class="hero-copy">Analyze a message in seconds with an NLP-powered spam classifier. Paste an email or SMS below and let the model assess whether it looks like spam.</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------- METRIC CARDS --------------------
st.markdown('<div class="section-label">Model overview</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "Reported accuracy", "97%", "Overall correct predictions", "↗"),
    (m2, "Reported precision", "1.00", "Spam predictions: reported precision", "◎"),
    (m3, "Text processing", "NLP", "Tokenization · stopwords · stemming", "⌘"),
    (m4, "Prediction mode", "Live", "Analyze one message at a time", "✳"),
]
for col, title, value, foot, icon in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-top"><span>{title}</span><span class="metric-icon">{icon}</span></div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot">{foot}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">Message analysis</div>
      <div class="panel-desc">Paste the content you want to check. Your text is processed by the loaded model.</div>
    </div>
    """, unsafe_allow_html=True)
    message = st.text_area(
        "Message content",
        placeholder="Example: Congratulations! You have won a prize. Click here to claim your reward...",
        height=210,
        label_visibility="collapsed",
        key="message_input",
    )
    count_col, hint_col = st.columns([1, 2])
    with count_col:
        st.caption(f"{len(message)} characters")
    with hint_col:
        st.caption("Tip: Include the full message for a more useful prediction.")
    predict = st.button("✦  Analyze message", use_container_width=True)

    if predict:
        if not message.strip():
            st.warning("Please enter a message before running the analysis.")
        else:
            try:
                transformed = transform_text(message, stop_words, ps)
                vector_input = tfidf.transform([transformed])
                prediction = int(model.predict(vector_input)[0])
                st.session_state["last_result"] = {
                    "prediction": prediction,
                    "message": message,
                    "transformed": transformed,
                }
            except Exception as exc:
                st.error("Could not analyze this message. Check the NLTK resources and model files.")
                st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        if result["prediction"] == 1:
            st.markdown("""
            <div class="result-card result-spam">
              <div class="result-label">Analysis result</div>
              <div class="result-main" style="color:#FF8FA0">⚠ &nbsp; Spam detected</div>
              <div class="result-note">The model classified this message as spam. Be cautious with unexpected links, requests for personal information, or urgent payment claims.</div>
              <span class="result-badge badge-spam">SPAM CLASSIFICATION</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-safe">
              <div class="result-label">Analysis result</div>
              <div class="result-main" style="color:#75E8D6">✓ &nbsp; Not Spam</div>
              <div class="result-note">The model classified this message as not spam, but for security, please also check the senders.</div>
              <span class="result-badge badge-safe">NON-SPAM CLASSIFICATION</span>
            </div>
            """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">How it works</div>
      <div class="panel-desc">A compact NLP pipeline turns raw text into a model prediction.</div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">01 · Normalize text</div><div class="info-copy">Convert the message to lowercase and tokenize its words.</div></div></div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">02 · Clean tokens</div><div class="info-copy">Remove non-alphanumeric tokens and common English stopwords.</div></div></div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">03 · Stem words</div><div class="info-copy">Reduce words to their stems using the Porter Stemmer.</div></div></div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">04 · Vectorize & classify</div><div class="info-copy">TF-IDF converts the text into features that the trained model uses for its prediction.</div></div></div>
    </div>
    <div style="height:16px"></div>
    <div class="panel">
      <div class="panel-title">Performance notes</div>
      <div class="panel-desc">Metrics should be read in the context of the test dataset.</div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">97% reported accuracy</div><div class="info-copy">The reported share of predictions that were correct.</div></div></div>
      <div class="info-row"><div class="info-dot"></div><div><div class="info-heading">1.00 reported precision</div><div class="info-copy">If this is spam-class precision, it means no false positives in the evaluation set. Confirm the class label and test split before publishing.</div></div></div>
      <div style="border-top:1px solid rgba(157,178,211,.14);padding-top:13px;margin-top:12px;font-size:11px;color:#7E8DA5;line-height:1.7">
        Note: This is a text classifier, not a full email security scanner. Never treat a non-spam result as a safety guarantee.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""

<span>
  Generated by
  <a href="https://www.linkedin.com/in/m-zainkhan/"
     target="_blank"
     rel="noopener noreferrer"
     style="color:#63E6D3; text-decoration:none; font-weight:700;">
    Zain Khan ↗
  </a>
</span>
<div class="footer">
  <span>MAILGUARD · MACHINE LEARNING PROJECT</span>
  <span>TF-IDF + trained classifier · Streamlit UI</span>
</div>
""", unsafe_allow_html=True)
