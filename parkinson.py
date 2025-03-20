import numpy as np
import pandas as pd
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sounddevice as sd
import soundfile as sf
import datetime
import time
import streamlit as st

# Define feature names based on the Parkinson's dataset
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
    'spread1', 'spread2', 'D2', 'PPE'
]

# Function to extract features from audio
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    features = []
    
    # Fundamental frequency (Fo, Fhi, Flo)
    f0 = librosa.yin(y, fmin=50, fmax=500)
    features.append(np.mean(f0))  # MDVP:Fo(Hz)
    features.append(np.max(f0))   # MDVP:Fhi(Hz)
    features.append(np.min(f0))   # MDVP:Flo(Hz)
    
    # Jitter measures
    jitter = (np.std(f0) / np.mean(f0)) * 100  # Simplified jitter (%)
    features.append(jitter)  # MDVP:Jitter(%)
    features.append(np.std(f0) / sr)  # MDVP:Jitter(Abs)
    features.append(jitter / 2)  # MDVP:RAP (approximation)
    features.append(jitter / 3)  # MDVP:PPQ (approximation)
    features.append(jitter * 1.5)  # Jitter:DDP (approximation)
    
    # Shimmer measures
    shimmer = np.std(np.abs(y))  # Simplified shimmer
    features.append(shimmer)  # MDVP:Shimmer
    features.append(10 * np.log10(shimmer + 1e-10))  # MDVP:Shimmer(dB)
    features.append(shimmer / 2)  # Shimmer:APQ3 (approximation)
    features.append(shimmer / 1.5)  # Shimmer:APQ5 (approximation)
    features.append(shimmer * 0.8)  # MDVP:APQ (approximation)
    features.append(shimmer * 1.2)  # Shimmer:DDA (approximation)
    
    # NHR and HNR (simplified approximations)
    noise = np.random.normal(0, 0.01, len(y))
    nhr = np.mean(np.abs(noise)) / np.mean(np.abs(y))
    hnr = 10 * np.log10(np.mean(np.abs(y)) / (np.mean(np.abs(noise)) + 1e-10))
    features.append(nhr)  # NHR
    features.append(hnr)  # HNR
    
    # Nonlinear measures (approximated using MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.append(np.mean(mfccs[0]))  # RPDE (approximation)
    features.append(np.std(mfccs[0]))   # DFA (approximation)
    features.append(np.max(mfccs[1]))   # spread1 (approximation)
    features.append(np.std(mfccs[1]))   # spread2 (approximation)
    features.append(np.mean(mfccs[2]))  # D2 (approximation)
    features.append(np.max(mfccs[3]))   # PPE (approximation)
    
    return np.array(features).reshape(1, -1)

# Function to train or load the model
def train_or_load_model(dataset_path='parkinsons.data'):
    model_path = 'parkinsons_model.pkl'
    scaler_path = 'scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.info("Loaded existing model and scaler.")
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Dataset not found. Please provide parkinsons.data.")
        
        data = pd.read_csv(dataset_path)
        X = data.drop(['name', 'status'], axis=1)
        y = data['status']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        st.success("Model trained and saved.")
    
    return model, scaler

# Function to predict Parkinson's from audio
def predict_parkinsons(audio_file, model, scaler):
    features = extract_features(audio_file)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction[0], probability, features_scaled

# Function to record audio
def record_audio(duration=5, sample_rate=22050):
    st.write(f"Recording will start in 3 seconds and last for {duration} seconds...")
    for i in range(3, 0, -1):
        st.write(f"{i}...")
        time.sleep(1)
    
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording finished!")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    sf.write(filename, recording, sample_rate)
    st.success(f"Saved recording as: {filename}")
    return filename

# Function to identify symptoms in human-understandable terms
def identify_symptoms(features_scaled, feature_names):
    symptoms = []
    thresholds = {
        'MDVP:Jitter(%)': 0.5,  # High jitter indicates vocal instability
        'MDVP:Shimmer': 0.5,    # High shimmer indicates amplitude variation
        'NHR': 0.3,             # High noise-to-harmonics ratio
        'HNR': -0.5,            # Low harmonics-to-noise ratio
        'PPE': 0.5              # High pitch period entropy
    }
    
    for i, (feature_value, feature_name) in enumerate(zip(features_scaled[0], feature_names)):
        if feature_name in thresholds:
            threshold = thresholds[feature_name]
            if (feature_name in ['HNR'] and feature_value < threshold) or \
               (feature_name not in ['HNR'] and feature_value > threshold):
                if feature_name == 'MDVP:Jitter(%)':
                    symptoms.append("Shaky or trembling voice (voice sounds unsteady)")
                elif feature_name == 'MDVP:Shimmer':
                    symptoms.append("Voice sounds weak or breathy (volume changes a lot)")
                elif feature_name == 'NHR':
                    symptoms.append("Hoarse or rough voice (sounds scratchy)")
                elif feature_name == 'HNR':
                    symptoms.append("Voice sounds unclear or muffled (hard to understand)")
                elif feature_name == 'PPE':
                    symptoms.append("Voice pitch changes unexpectedly (sounds uneven)")
    
    return symptoms if symptoms else ["No noticeable voice or speech issues detected"]

# Streamlit app
def main():
    st.title("Parkinson's Disease Early Detection")
    st.write("This app analyzes voice recordings to detect early signs of Parkinson's disease using machine learning.")

    dataset_path = 'parkinsons.data'
    
    if not os.path.exists(dataset_path):
        st.warning("Dataset 'parkinsons.data' not found. Please place it in the current directory to train the model.")
        st.warning("You can still use a pre-trained model if available.")
    
    try:
        model, scaler = train_or_load_model(dataset_path)
    except FileNotFoundError as e:
        if os.path.exists('parkinsons_model.pkl') and os.path.exists('scaler.pkl'):
            model, scaler = train_or_load_model()
        else:
            st.error("No pre-trained model found and dataset is missing. Please provide the dataset or pre-trained model files.")
            return

    option = st.sidebar.selectbox("Choose an option", ["Upload Audio File", "Record Audio"])

    if option == "Upload Audio File":
        st.subheader("Upload an Audio File")
        uploaded_file = st.file_uploader("Choose an audio file (e.g., WAV format)", type=["wav"])
        
        if uploaded_file is not None:
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio("temp_audio.wav", format="audio/wav")
            
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    try:
                        prediction, probability, features_scaled = predict_parkinsons("temp_audio.wav", model, scaler)
                        result = "Parkinson's disease detected (early stage possible)" if prediction == 1 else "No Parkinson's disease detected"
                        symptoms = identify_symptoms(features_scaled, FEATURE_NAMES)
                        
                        # Display results with title-like formatting
                        st.markdown("### Analysis Results")
                        st.markdown("#### Detection")
                        st.write(result)
                        st.markdown("#### Confidence")
                        st.write(f"{probability * 100:.2f}%")
                        st.markdown("#### Voice and Speech Observations")
                        for symptom in symptoms:
                            st.write(f"- {symptom}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                os.remove("temp_audio.wav")

    elif option == "Record Audio":
        st.subheader("Record Your Voice")
        duration = st.slider("Select recording duration (seconds)", 1, 10, 5)
        
        if st.button("Start Recording"):
            with st.spinner("Recording..."):
                try:
                    audio_file = record_audio(duration=duration)
                    st.audio(audio_file, format="audio/wav")
                    
                    st.write("Analyzing recording...")
                    prediction, probability, features_scaled = predict_parkinsons(audio_file, model, scaler)
                    result = "Parkinson's disease detected (early stage possible)" if prediction == 1 else "No Parkinson's disease detected"
                    symptoms = identify_symptoms(features_scaled, FEATURE_NAMES)
                    
                    # Display results with title-like formatting
                    st.markdown("### Analysis Results")
                    st.markdown("#### Detection")
                    st.write(result)
                    st.markdown("#### Confidence")
                    st.write(f"{probability * 100:.2f}%")
                    st.markdown("#### Voice and Speech Observations")
                    for symptom in symptoms:
                        st.write(f"- {symptom}")
                except Exception as e:
                    st.error(f"An error occurred during recording or analysis: {e}")

if __name__ == "__main__":
    main()