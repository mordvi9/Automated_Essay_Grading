import streamlit as st
import pandas as pd
import joblib
from pathlib import Path # <--- Import Path
import sys


try:
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    ROOT_DIR = Path.cwd()

sys.path.append(str(ROOT_DIR / 'src'))

from feature_extractor import load_nlp_models, extract_all_features


st.set_page_config(
    page_title="Automated Essay Scorer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    
    
    /* Main Title */
    h1 {
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        font-weight: bold;
    }

    /* Sub-headers */
    h2, h3 {
        color: #1F2937; /* Dark Gray */
    }

    /* Input Labels */
    .stTextArea label, .stTextInput label {
        font-weight: bold;
        color: #374151; /* Medium Gray */
        font-size: 1.1rem;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #2563EB; /* Bright Blue */
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8; /* Darker Blue on hover */
    }

    /* Metric Card for Score */
    div[data-testid="stMetric"] {
        border: 1px solid #D1D5DB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    print("Loading inference artifacts...")
    artifacts_path = ROOT_DIR / "models" / "aes_full_pipeline_artifacts.pkl"
    artifacts = joblib.load(artifacts_path)
    nlp, bert_tokenizer, bert_model = load_nlp_models()
    return artifacts, nlp, bert_tokenizer, bert_model

artifacts, nlp, tokenizer, model = load_all()
pipeline = artifacts['pipeline']
regressor = artifacts['regressor']
tfidf_vectorizer = artifacts['tfidf_vectorizer']
training_columns = artifacts['training_columns']


with st.sidebar:
    st.title("✍️ About the Project")
    st.info("This demo uses a regression model to predict essay scores based on the IELTS scoring rubric. "
             "It analyzes linguistic complexity, semantic relevance, and grammatical structure.")
    st.markdown("---")
    st.subheader("Technology Stack")
    st.markdown("""
    - **Backend:** Python, Scikit-learn, AutoML, XGBoost
    - **NLP:** SpaCy, Hugging Face Transformers
    - **Frontend:** Streamlit
    """)
    st.markdown("---")
    st.link_button("View on GitHub", "https://github.com/mordvi9/Automated_Essay_Grading")
    st.markdown("---")

st.title("📝 Automated Essay Scoring Engine")
st.write("Enter an essay prompt and the corresponding essay below, then click 'Score My Essay' to get an instant evaluation.")
st.divider()

# Use columns for a clean side-by-side layout
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Prompt")
    prompt = st.text_area("Paste the prompt here:", height=200, label_visibility="collapsed")

with col2:
    st.subheader("Essay")
    essay = st.text_area("Paste your essay here:", height=400, label_visibility="collapsed")

st.divider()

_, button_col, _ = st.columns([3, 2, 3])
if button_col.button("Score My Essay"):
    if not essay.strip():
        st.error("Please enter an essay to score.")
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("⏳ Analyzing essay and calculating score..."):
            feature_dict = extract_all_features(prompt, essay, tfidf_vectorizer, nlp, tokenizer, model)
            inference_df = pd.DataFrame([feature_dict])
            transformed_features = pipeline.transform(inference_df)
            predicted_score = regressor.predict(transformed_features)[0]
            
        st.subheader("Evaluation Result")
        score_col, details_col = st.columns([1, 2], gap="large")
        
        with score_col:
            st.metric(label="Predicted Score", value=f"{predicted_score:.2f}")

        with details_col:
            st.write("**Key Feature Analysis:**")
            st.json({
                "Word Count": len(essay.split()),
                "Avg. Sentence Length": f"{feature_dict.get('avg_sentence_length', 0):.2f}",
                "Grammar Issues": f"{feature_dict.get('grammar_error_count', 0)}",
                "Prompt Relevance": f"{feature_dict.get('bert_cosine_similarity', 0):.3f}"
            })
        