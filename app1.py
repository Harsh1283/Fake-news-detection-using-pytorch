import streamlit as st
import torch
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.serialization import safe_globals
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the model architecture
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x

def load_model_and_vectorizer():
    try:
        # Load the vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Load the pre-trained model
        model = FakeNewsClassifier(input_dim=5000)
        model.load_state_dict(torch.load('fakenews_model.pt'))
        model.eval()  # Set to evaluation mode
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {str(e)}")
        return None, None

def predict_news(text, model, vectorizer):
    with torch.no_grad():
        text_vectorized = vectorizer.transform([text]).toarray()
        text_tensor = torch.FloatTensor(text_vectorized)
        output = model(text_tensor)
        prediction = torch.round(output).item()
        confidence = output.item()
        return "REAL" if prediction == 1 else "FAKE", confidence

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Run Model", "About", "Model Performance"])

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

if page == "Home":
    st.title("🔍 Fake News Detection System")
    st.write("""
    ## Welcome to the Fake News Detection System
    This application uses advanced machine learning to help identify potential fake news articles.
    
    ### Features:
    - Real-time news analysis
    - High accuracy prediction
    - Detailed performance metrics
    - User-friendly interface
    
    Get started by navigating to the 'Run Model' section!
    """)
    
    # Add some sample statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Accuracy", value="94.5%")
    with col2:
        st.metric(label="Training Samples", value="50K+")
    with col3:
        st.metric(label="Processing Time", value="<1 sec")

elif page == "Run Model":
    st.title("📊 Analyze News")
    
    # Text input
    news_text = st.text_area("Enter news text to analyze:", height=150)
    
    if st.button("Analyze"):
        if news_text and model and vectorizer:
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_news(news_text, model, vectorizer)
                
                # Display result with custom styling
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == "REAL":
                        st.success("Prediction: REAL NEWS")
                        gauge_color = "green"
                    else:
                        st.error("Prediction: FAKE NEWS")
                        gauge_color = "red"
                with col2:
                    st.info(f"Confidence: {confidence*100:.2f}%")
                
                # Enhanced confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': "Prediction Confidence"},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                # Update layout for better visualization
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font={'size': 16}
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.title("ℹ️ About the Project")
    st.write("""
    ### Project Overview
    This fake news detection system was developed using PyTorch and employs 
    state-of-the-art natural language processing techniques.
    
    ### How it Works
    1. Text Preprocessing
    2. TF-IDF Vectorization
    3. Neural Network Classification
    4. Confidence Score Generation
    
    ### Team
    - Developer: [Your Name]
    - Supervisor: [Supervisor Name]
    - Institution: [Your Institution]
    """)

elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    
    # Sample metrics (replace with your actual metrics)
    metrics = {
        'Accuracy': 0.945,
        'Precision': 0.932,
        'Recall': 0.956,
        'F1 Score': 0.944
    }
    
    # Metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics bar chart
        fig = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title="Model Metrics"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Sample confusion matrix
        conf_matrix = [[450, 50],
                      [25, 475]]
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")
st.sidebar.markdown("Model Status: **Active**")
st.sidebar.markdown("Last Updated: **2024-04-29**")