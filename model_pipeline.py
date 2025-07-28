import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
from joblib import load 
import plotly.express as px
import plotly.graph_objects as go

# page config 
st.set_page_config(
    page_title="🎧 Podcast Ad Detector",
    page_icon="🎧",
    layout="wide"
)


# Load model
@st.cache_resource
def load_model():
    try:
        with open('feature_analysis/models/xgb_best_model_20250727.joblib', 'rb') as f:
            return load(f)
    except:
        st.error("Model file not found! Please upload your model joblib")
        return None


def extract_segment_features(audio_file, start_time, duration):
    
    # Load audio segment
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration, sr=22050)
    
    if len(y) == 0:
        return None 
    
    features = {}
    
    # ==========================================
    # SPECTRAL FEATURES (Frequency Domain)
    # ==========================================
    
    # 1. Spectral Centroid - "brightness" of sound
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    # 2. Spectral Bandwidth - "width" of frequency spectrum
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # 3. Spectral Rolloff - frequency below which 85% of energy is contained
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # 4. Spectral Contrast - difference between peaks and valleys in spectrum
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # 5. Spectral Flatness - measure of how "noise-like" vs "tone-like" the sound is
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    
    # ==========================================
    # TEMPORAL FEATURES (Time Domain)
    # ==========================================
    
    # 6. RMS Energy - overall loudness/energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    
    # 7. Zero Crossing Rate - how often signal crosses zero (roughness indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    
    # 8. Basic statistical features
    features['amplitude_max'] = np.max(np.abs(y))
    
    # ==========================================
    # TIMBRAL FEATURES (MFCC - Speech Characteristics)
    # ==========================================
    
    # 9. MFCCs - Mel-frequency cepstral coefficients (speech/music characteristics)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    del features['mfcc_2_mean']
    # ==========================================
    # RHYTHMIC FEATURES (Tempo/Beat)
    # ==========================================
    
    # 10. Tempo estimation
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['beat_count'] = len(beats)
        features['beats_per_second'] = len(beats) / duration
    except:
        features['tempo'] = 0
        features['beat_count'] = 0
        features['beats_per_second'] = 0
    
    # ==========================================
    # HARMONIC FEATURES (Pitch/Harmony)
    # ==========================================
    
    # 11. Chroma features - pitch class profiles
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
    except:
        features['chroma_mean'] = 0
        features['chroma_std'] = 0
    
    # 12. Tonnetz - harmonic network features
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
    except:
        features['tonnetz_mean'] = 0
        features['tonnetz_std'] = 0
    
    # ==========================================
    # ADVANCED FEATURES
    # ==========================================
    
    # 13. Mel-scale Spectrogram statistics
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spec_mean'] = np.mean(mel_spec_db)
    features['mel_spec_std'] = np.std(mel_spec_db)
    
    return features

def analyze_podcast(audio_file_path, segment_duration=30):
    """Analyze podcast and return results"""
    model = load_model()
    if model is None:
        return None, None
    
    # Get duration
    y_full, sr = librosa.load(audio_file_path, sr=22050)
    total_duration = len(y_full) / sr
    
    # Create segments
    segments = []
    current_time = 0
    
    while current_time < total_duration:
        segment_end = min(current_time + segment_duration, total_duration)
        actual_duration = segment_end - current_time
        
        if actual_duration < 5:
            break
            
        segments.append({
            'start_time': current_time,
            'end_time': segment_end,
            'duration': actual_duration
        })
        
        current_time += segment_duration
    
    # Analyze segments
    results = []
    progress_bar = st.progress(0)
    
    for i, segment in enumerate(segments):
        progress_bar.progress((i + 1) / len(segments))
        
        features = extract_segment_features(
            audio_file_path, 
            segment['start_time'], 
            segment['duration']
        )
        
        if features is None:
            continue
            
        feature_df = pd.DataFrame([features])
        prediction_proba = model.predict_proba(feature_df)[0] # converts from 2D array to 1D array 
        
        # Get probability of ad (class 1)
        ad_probability = prediction_proba[1] if len(prediction_proba) > 1 else 0
        
        # Apply custom threshold: >= 0.21 = ad, < 0.21 = content
        prediction = 'ad' if ad_probability >= 0.21 else 'content'
        confidence = ad_probability if prediction == 'ad' else (1 - ad_probability)
        
        results.append({
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['duration'],
            'prediction': prediction,
            'confidence': confidence,
            'ad_probability': ad_probability
        })
    
    progress_bar.empty()
    return results, total_duration

# Main app
def main():
    st.title("🎧 Podcast Ad Detector")
    st.markdown("Upload your podcast and see which segments are likely ads!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your podcast file",
        type=['mp3', 'wav', 'm4a'],
        help="Upload an audio file to analyze for advertisements"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Analyze for Ads", type="primary"):
            with st.spinner("Processing your podcast... This may take a few minutes."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Analyze
                results, total_duration = analyze_podcast(tmp_file_path)
                
                if results:
                    # Calculate stats
                    ad_segments = [r for r in results if r['prediction'] == 'ad']
                    ad_duration = sum([r['duration'] for r in ad_segments])
                    
                    # Display stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Duration", f"{total_duration/60:.1f} min")
                    with col2:
                        st.metric("Ad Segments", len(ad_segments))
                    with col3:
                        st.metric("Ad Duration", f"{ad_duration/60:.1f} min")
                    with col4:
                        st.metric("Ads Percentage", f"{(ad_duration/total_duration)*100:.1f}%")
                    
                    # Create timeline visualization
                    st.subheader("📊 Timeline Visualization")
                    
                    fig = go.Figure()
                    
                    for result in results:
                        color = '#FF6B6B' if result['prediction'] == 'ad' else '#4ECDC4'
                        fig.add_trace(go.Scatter(
                            x=[result['start_time']/60, result['end_time']/60],
                            y=[1, 1],
                            mode='lines',
                            line=dict(color=color, width=20),
                            name=result['prediction'].title(),
                            hovertemplate=f"<b>{result['prediction'].title()}</b><br>" +
                                        f"Time: {result['start_time']/60:.1f} - {result['end_time']/60:.1f} min<br>" +
                                        f"Confidence: {result['confidence']:.1%}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title="Podcast Timeline (Red = Ads, Teal = Content)",
                        xaxis_title="Time (minutes)",
                        yaxis=dict(visible=False),
                        height=200,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    with st.expander("📋 Detailed Results"):
                        df = pd.DataFrame(results)
                        df['start_min'] = df['start_time'] / 60
                        df['end_min'] = df['end_time'] / 60
                        df['confidence_pct'] = df['confidence'] * 100
                        
                        st.dataframe(
                            df[['start_min', 'end_min', 'prediction', 'confidence_pct']],
                            column_config={
                                'start_min': st.column_config.NumberColumn('Start (min)', format="%.1f"),
                                'end_min': st.column_config.NumberColumn('End (min)', format="%.1f"),
                                'prediction': st.column_config.TextColumn('Prediction'),
                                'confidence_pct': st.column_config.NumberColumn('Confidence (%)', format="%.1f")
                            }
                        )

if __name__ == "__main__":
    main()