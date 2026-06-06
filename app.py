import streamlit as st
import pickle
import numpy as np
import time

# Set up clean page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a beautiful, modern UI
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    </style>
""", unsafe_with_html=True)

# App Header
st.markdown("<div class='main-title'>🛡️ Advanced Meta-Ensemble Fake News Detection Platform</div>", unsafe_with_html=True)
st.markdown("---")

# Load the saved Pickle pipeline safely
@st.cache_resource
def load_model():
    try:
        with open('fake_news_ensemble_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

pipeline = load_model()

# Handle error if model script hasn't been run yet
if pipeline is None:
    st.error("**Model file not found!** Please run your training script (`train_model.py`) first to generate `fake_news_ensemble_model.pkl`.")
    st.stop()

# Sidebar Information for Reviewers/Professors
with st.sidebar:
    st.header("Project Architecture")
    st.markdown("""
    **MCA Minor Project**
    * **Features:** TF-IDF Vectorization
    * **Framework:** Soft Voting Meta-Ensemble
    
    **Engine Components:**
    1. Logistic Regression
    2. Random Forest
    3. Support Vector Machine (SVM)
    4. Multinomial Naive Bayes
    5. Gradient Boosting (GBM)
    """)

# User input text field
st.subheader("Analyze Content")
user_input = st.text_area(
    "Paste the full news article body or headline below:", 
    placeholder="Type or paste text here (minimum 10 words recommended for optimal accuracy)...",
    height=200
)

# Predict Button logic
if st.button("Run Ensemble Verification", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text patterns across all 5 models..."):
            # Simulate a small processing lag for aesthetic UX
            time.sleep(1.2)
            
            # Predict probability score and class
            # 0 = Fake, 1 = True
            prediction = pipeline.predict([user_input])[0]
            probabilities = pipeline.predict_proba([user_input])[0]
            
            # Extract specific confidences
            fake_confidence = probabilities[0] * 100
            true_confidence = probabilities[1] * 100

        st.markdown("### Verification Result")
        
        # Display custom styled alerts based on prediction
        if prediction == 1:
            st.success(f"### 🎉 **Verified Content (REAL NEWS)**")
            st.markdown(f"The ensemble engine is highly confident (**{true_confidence:.2f}%**) that this text matches legitimate reporting patterns.")
            st.balloons()
        else:
            st.error(f"### 🚨 **Deceptive Content Flagged (FAKE NEWS)**")
            st.markdown(f"The ensemble engine detected a **{fake_confidence:.2f}%** statistical likelihood of structural fabrication or misinformation traits.")

        # Beautiful visualization layout for the presentation review
        st.markdown("---")
        st.subheader("📊 Engine Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h5>Truth Score Index</h5>
                <h2>{true_confidence:.1f}%</h2>
                <p style='color:gray; font-size:12px;'>Probability calculated via averaged log-weights</p>
            </div>
            """, unsafe_with_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color: #EF4444;'>
                <h5>Deception Marker Index</h5>
                <h2>{fake_confidence:.1f}%</h2>
                <p style='color:gray; font-size:12px;'>Linguistic variances captured by the ensemble matrix</p>
            </div>
            """, unsafe_with_html=True)

# Footer 
st.markdown("<br><hr><center style='color:gray; font-size:12px;'>MCA Final Project Review Presentation App Framework • Built with Streamlit</center>", unsafe_with_html=True)