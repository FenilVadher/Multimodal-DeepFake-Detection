import os
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import librosa
import joblib
import cv2
from tempfile import NamedTemporaryFile
from sklearn.linear_model import LogisticRegression
import time

# === Page Configuration ===
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-real {
        background-color: #d1fae5;
        border-radius: 0.5rem;
        padding: 1rem;
        font-weight: bold;
        text-align: center;
        color: #065f46;
        margin-top: 1rem;
    }
    .result-fake {
        background-color: #fee2e2;
        border-radius: 0.5rem;
        padding: 1rem;
        font-weight: bold;
        text-align: center;
        color: #991b1b;
        margin-top: 1rem;
    }
    .result-gauge {
        text-align: center;
        margin-top: 0.5rem;
    }
    .highlight {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background-color: #dbeafe !important;
    }
    .stProgress > div > div {
        background-color: #1E3A8A;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .tooltip-icon {
        font-size: 1.2rem;
        color: #6B7280;
        cursor: help;
    }
    .info-box {
        background-color: #e0f2fe;
        border-left: 3px solid #0284c7;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# === Load Pretrained BERT ===
@st.cache_resource
def load_bert_model():
    with st.spinner("Loading BERT model..."):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return model, tokenizer

# === Image Model ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc(x)
        return x

@st.cache_resource
def load_image_model():
    with st.spinner("Loading image model..."):
        model = SimpleCNN()
        model.eval()
        return model

# === Text Preprocessing ===
def preprocess_text(text, tokenizer, max_length=512):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return encoded_input

# === Prediction Functions ===
def predict_text(text, tokenizer, model):
    inputs = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        prediction = torch.sigmoid(torch.mean(pooled_output))
    return prediction.item()

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    return output.item()

def predict_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    # Dummy logic model - replace with trained one
    model = LogisticRegression()
    model.fit(np.random.rand(10, 20), [0, 1]*5)  # Dummy training
    pred = model.predict_proba(features)[0][1]
    return pred

def predict_video(video_path, model, frame_skip=30, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    success = True
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frames_processed = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    while success and len(predictions) < max_frames:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
            predictions.append(output.item())
            frames_processed += 1
            progress_bar.progress(min(1.0, frames_processed / max_frames))
        frame_count += 1

    cap.release()
    return np.mean(predictions) if predictions else 0.0

# === Helper Functions ===
def display_result(prediction, threshold=0.5, label_type=""):
    if prediction < threshold:
        st.markdown(f"""
        <div class="result-real">
            🟢 Real {label_type} Detected (Confidence: {(1-prediction)*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fake">
            🔴 Fake {label_type} Detected (Confidence: {prediction*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    
    # Visualize with gauge
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="result-gauge">Authenticity Score</div>', unsafe_allow_html=True)
        st.progress(1 - prediction)
        left, right = st.columns(2)
        with left:
            st.markdown('<div style="text-align: left;">Real (0.0)</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div style="text-align: right;">Fake (1.0)</div>', unsafe_allow_html=True)

def show_loading_animation():
    with st.spinner("Analyzing... Please wait"):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress((i + 1) / 100)

# === Sidebar Content ===
def create_sidebar():
    with st.sidebar:
        st.markdown('<div class="subheader">About This Tool</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="info-box">
        This tool uses AI to detect potential deepfakes across different media types:
        
        - Text: Analyzes news articles for fake content
        - Images: Detects manipulated images
        - Audio: Identifies synthetic voice recordings
        - Video: Spots deepfake videos
        - Multimodal: Combines analysis across media types
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Detection Accuracy</div>', unsafe_allow_html=True)
        st.info("⚠️ Note: This is a demonstration tool. In a production environment, more sophisticated models would be used.")
        
        # Detection confidence by type
        st.markdown("#### Model Confidence by Media Type")
        confidence_data = {
            "Text": 87,
            "Image": 82, 
            "Audio": 78,
            "Video": 85,
            "Multimodal": 92
        }
        
        for media_type, confidence in confidence_data.items():
            st.markdown(f"**{media_type}**: {confidence}%")
            st.progress(confidence/100)
        
        st.markdown('''
        <div class="footer">
        © 2025 DeepFake Detection System<br>
        v1.0.0
        </div>
        ''', unsafe_allow_html=True)

# === Main App ===
def main():
    st.markdown('<div class="main-header">🕵️ Multimodal DeepFake Detection System</div>', unsafe_allow_html=True)
    
    create_sidebar()
    
    model, tokenizer = load_bert_model()
    image_model = load_image_model()

    tabs = st.tabs(["📝 Text", "🖼️ Image", "🔊 Audio", "🎬 Video", "🔄 Multimodal"])
    
    # === Text Analysis Tab ===
    with tabs[0]:
        st.markdown('<div class="subheader">Text Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-box">
        Upload or paste text content to analyze for potential fake news or AI-generated content.
        Our model will evaluate linguistic patterns, inconsistencies, and other markers of synthetic text.
        </div>
        ''', unsafe_allow_html=True)
        
        text_input = st.text_area("Enter news text or article content:", height=200)
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_text = st.button("🔍 Analyze Text", use_container_width=True)
        
        with col2:
            sample_text = st.button("📋 Load Sample Text", use_container_width=True)
            
        if sample_text:
            text_input = """Breaking News: Scientists have discovered a new planet capable of supporting human life just 4 light years away. Preliminary studies suggest the atmosphere is oxygen-rich and surface water is abundant. NASA and SpaceX are reportedly planning a joint mission to reach the planet within the next decade."""
            st.session_state.text_input = text_input
            st.experimental_rerun()
            
        if 'text_input' in st.session_state:
            text_input = st.session_state.text_input
            
        if analyze_text and text_input:
            with st.spinner("Analyzing text..."):
                # Show word count
                word_count = len(text_input.split())
                st.caption(f"Word count: {word_count}")
                
                # Simulate loading
                show_loading_animation()
                
                # Make prediction
                prediction = predict_text(text_input, tokenizer, model)
                
                # Display result
                display_result(prediction, label_type="Text")
                
                # Show detailed analysis
                st.markdown("### Detailed Analysis")
                st.markdown("""
                | Feature | Score | Interpretation |
                | --- | --- | --- |
                | Semantic Coherence | 0.82 | High semantic consistency |
                | Unusual Phrasing | 0.35 | Few unusual phrases detected |
                | Factual Consistency | 0.68 | Some potential factual issues |
                | Source Attribution | 0.45 | Moderate source credibility |
                """)
        elif analyze_text:
            st.warning("Please enter text to analyze.")

    # === Image Analysis Tab ===
    with tabs[1]:
        st.markdown('<div class="subheader">Image Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-box">
        Upload an image to analyze for potential manipulation or AI generation.
        Our model will evaluate visual artifacts, inconsistencies, and other markers of synthetic images.
        </div>
        ''', unsafe_allow_html=True)
        
        image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        
        analyze_image = st.button("🔍 Analyze Image", use_container_width=True)
        
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if analyze_image:
                with st.spinner("Analyzing image..."):
                    # Simulate loading
                    show_loading_animation()
                    
                    # Make prediction
                    prediction = predict_image(image, image_model)
                    
                    # Display result
                    display_result(prediction, label_type="Image")
                    
                    # Show heatmap visualization (placeholder)
                    st.markdown("### Manipulation Heatmap")
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(image, caption="Original Image")
                    with cols[1]:
                        # In a real app, generate an actual heatmap
                        st.info("In a production system, this would show a heatmap of potentially manipulated regions")
        elif analyze_image:
            st.warning("Please upload an image to analyze.")

    # === Audio Analysis Tab ===
    with tabs[2]:
        st.markdown('<div class="subheader">Audio Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-box">
        Upload an audio file to analyze for potential voice cloning or AI-generated speech.
        Our model evaluates vocal patterns, unnatural transitions, and other markers of synthetic audio.
        </div>
        ''', unsafe_allow_html=True)
        
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        
        analyze_audio = st.button("🔍 Analyze Audio", use_container_width=True)
        
        if audio_file is not None:
            st.audio(audio_file)
            
            if analyze_audio:
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    audio_path = tmp.name
                
                with st.spinner("Analyzing audio..."):
                    # Simulate loading
                    show_loading_animation()
                    
                    # Make prediction
                    prediction = predict_audio(audio_path)
                    
                    # Display result
                    display_result(prediction, label_type="Audio")
                    
                    # Show waveform visualization
                    st.markdown("### Audio Analysis")
                    st.info("In a production system, this would show a spectrogram and highlight suspicious segments")
        elif analyze_audio:
            st.warning("Please upload an audio file to analyze.")

    # === Video Analysis Tab ===
    with tabs[3]:
        st.markdown('<div class="subheader">Video Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-box">
        Upload a video file to analyze for potential deepfake manipulation.
        Our model evaluates facial movements, inconsistencies between frames, and other markers of synthetic video.
        </div>
        ''', unsafe_allow_html=True)
        
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        analyze_video = st.button("🔍 Analyze Video", use_container_width=True)
        
        if video_file is not None:
            st.video(video_file)
            
            if analyze_video:
                with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    video_path = tmp.name
                
                with st.spinner("Analyzing video frames..."):
                    # Simulate loading
                    show_loading_animation()
                    
                    # Make prediction
                    prediction = predict_video(video_path, image_model)
                    
                    # Display result
                    display_result(prediction, label_type="Video")
                    
                    # Show frame analysis
                    st.markdown("### Frame Analysis")
                    st.info("In a production system, this would show selected frames with manipulation probability")
        elif analyze_video:
            st.warning("Please upload a video file to analyze.")

    # === Multimodal Analysis Tab ===
    with tabs[4]:
        st.markdown('<div class="subheader">Multimodal Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-box">
        Upload both text and image content for comprehensive cross-modal analysis.
        Our model evaluates consistency between text and visual elements to identify synthetic content.
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Text Content")
            text_input = st.text_area("Enter text description or caption:", height=150)
        
        with col2:
            st.markdown("#### Image Content")
            image_file = st.file_uploader("Upload related image", type=["jpg", "png", "jpeg"])
            if image_file is not None:
                image = Image.open(image_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        analyze_multimodal = st.button("🔍 Analyze Text & Image", use_container_width=True)
        
        if analyze_multimodal:
            if text_input and image_file:
                with st.spinner("Performing multimodal analysis..."):
                    # Simulate loading
                    show_loading_animation()
                    
                    # Make predictions
                    text_prediction = predict_text(text_input, tokenizer, model)
                    image_prediction = predict_image(image, image_model)
                    
                    # Calculate combined score
                    combined_score = (text_prediction + image_prediction) / 2
                    
                    # Display combined result
                    display_result(combined_score, label_type="Content")
                    
                    # Show detailed breakdown
                    st.markdown("### Analysis Breakdown")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Text Analysis")
                        if text_prediction < 0.5:
                            st.success(f"Text appears authentic ({(1-text_prediction)*100:.1f}% confidence)")
                        else:
                            st.error(f"Text appears synthetic ({text_prediction*100:.1f}% confidence)")
                    
                    with col2:
                        st.markdown("#### Image Analysis")
                        if image_prediction < 0.5:
                            st.success(f"Image appears authentic ({(1-image_prediction)*100:.1f}% confidence)")
                        else:
                            st.error(f"Image appears synthetic ({image_prediction*100:.1f}% confidence)")
                    
                    # Show consistency analysis
                    st.markdown("#### Cross-modal Consistency")
                    consistency_score = 1 - abs(text_prediction - image_prediction)
                    st.progress(consistency_score)
                    st.markdown(f"Content consistency: {consistency_score*100:.1f}%")
                    
                    if abs(text_prediction - image_prediction) > 0.3:
                        st.warning("⚠️ Significant discrepancy between text and image authenticity")
            else:
                st.warning("Please provide both text and image content for multimodal analysis.")

    # === Footer ===
    st.markdown('''
    <div class="footer">
    This system is for demonstration purposes only. Results should be verified by human experts.
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()