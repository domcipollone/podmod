import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import joblib 
import plotly.express as px
import plotly.graph_objects as go
import requests
import feedparser
import os
import base64
from pydub import AudioSegment
import soundfile as sf
from urllib.parse import urljoin
import whisper
import re
from typing import List, Dict, Tuple

# Page config 
st.set_page_config(
    page_title="🎧 Podcast Ad Detector",
    page_icon="🎧",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    """Load both the XGBoost model and Whisper model"""
    xgb_model = joblib.load('./feature_analysis/models/xgb_best_model_20250727.joblib')
    whisper_model = whisper.load_model("base")  # You can use "small", "medium", "large" for better accuracy
    return xgb_model, whisper_model

def extract_segment_features(audio_file, start_time, duration):
    """Extract comprehensive features from a single audio segment"""
    
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
    except:
        features['tempo'] = 0
        features['beat_count'] = 0
    
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

def transcribe_and_segment(audio_file_path, whisper_model) -> List[Dict]:
    """Transcribe audio and segment into paragraphs with timestamps"""
    
    # Transcribe with Whisper
    result = whisper_model.transcribe(
        audio_file_path,
        language="en",
        verbose=False,
        word_timestamps=True  # This gives us word-level timestamps
    )
    
    # Extract segments (Whisper provides these automatically)
    segments = result['segments']
    
    # Group segments into paragraphs based on pauses
    paragraphs = []
    current_paragraph = {
        'text': '',
        'start': None,
        'end': None,
        'words': []
    }
    
    for segment in segments:
        # Check if this should start a new paragraph
        # (based on pause duration or punctuation)
        if current_paragraph['text'] and segment['start'] - current_paragraph['end'] > 1.5:
            # Save current paragraph
            paragraphs.append(current_paragraph)
            current_paragraph = {
                'text': '',
                'start': None,
                'end': None,
                'words': []
            }
        
        # Add to current paragraph
        if current_paragraph['start'] is None:
            current_paragraph['start'] = segment['start']
        
        current_paragraph['text'] += ' ' + segment['text'].strip()
        current_paragraph['end'] = segment['end']
        
        # Store word-level timestamps if available
        if 'words' in segment:
            current_paragraph['words'].extend(segment['words'])
    
    # Don't forget the last paragraph
    if current_paragraph['text']:
        paragraphs.append(current_paragraph)
    
    # Further split long paragraphs
    final_paragraphs = []
    for para in paragraphs:
        # If paragraph is too long (>30 seconds), split it
        if para['end'] - para['start'] > 30:
            # Split based on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para['text'].strip())
            
            if len(sentences) > 1 and para['words']:
                # Distribute sentences based on word timestamps
                word_idx = 0
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    
                    if word_idx < len(para['words']):
                        start_time = para['words'][word_idx]['start'] if word_idx < len(para['words']) else para['start']
                        end_idx = min(word_idx + sentence_words, len(para['words']) - 1)
                        end_time = para['words'][end_idx]['end'] if end_idx < len(para['words']) else para['end']
                        
                        final_paragraphs.append({
                            'text': sentence,
                            'start': start_time,
                            'end': end_time
                        })
                        
                        word_idx += sentence_words
            else:
                # Keep as is if can't split well
                final_paragraphs.append(para)
        else:
            final_paragraphs.append(para)
    
    return final_paragraphs

def analyze_podcast_with_paragraphs(audio_file_path):
    """Analyze podcast using paragraph-level segmentation"""
    xgb_model, whisper_model = load_models()
    
    if xgb_model is None or whisper_model is None:
        return None, None, None
    
    # Get total duration
    y_full, sr = librosa.load(audio_file_path, sr=22050)
    total_duration = len(y_full) / sr
    
    # Step 1: Transcribe and get paragraph segments
    with st.spinner("Transcribing audio and detecting paragraphs..."):
        paragraphs = transcribe_and_segment(audio_file_path, whisper_model)
    
    st.success(f"Found {len(paragraphs)} paragraphs to analyze")
    
    # Step 2: Analyze each paragraph
    results = []
    progress_bar = st.progress(0)
    
    for i, paragraph in enumerate(paragraphs):
        progress_bar.progress((i + 1) / len(paragraphs))
        
        # Extract features for this paragraph's audio segment
        duration = paragraph['end'] - paragraph['start']
        
        # Skip very short segments
        if duration < 1.0:
            continue
        
        features = extract_segment_features(
            audio_file_path, 
            paragraph['start'], 
            duration
        )
        
        if features is None:
            continue
        
        feature_df = pd.DataFrame([features])
        
        # Fix tempo processing if needed
        if 'tempo' in feature_df.columns:
            if feature_df['tempo'].dtype == 'object':
                feature_df['tempo'] = pd.to_numeric(feature_df['tempo'], errors='coerce').fillna(0)
        
        prediction_proba = xgb_model.predict_proba(feature_df)[0]
        
        # Get probability of ad (class 1)
        ad_probability = prediction_proba[1] if len(prediction_proba) > 1 else 0
        
        # Apply custom threshold: >= 0.21 = ad, < 0.21 = content
        prediction = 'ad' if ad_probability >= 0.21 else 'content'
        confidence = ad_probability if prediction == 'ad' else (1 - ad_probability)
        
        results.append({
            'start_time': paragraph['start'],
            'end_time': paragraph['end'],
            'duration': duration,
            'prediction': prediction,
            'confidence': confidence,
            'ad_probability': ad_probability,
            'text': paragraph['text'][:100] + '...' if len(paragraph['text']) > 100 else paragraph['text']
        })
    
    progress_bar.empty()
    
    # Merge consecutive segments with same prediction
    merged_results = []
    if results:
        current_segment = results[0].copy()
        
        for result in results[1:]:
            # If same prediction and consecutive, merge
            if (result['prediction'] == current_segment['prediction'] and 
                abs(result['start_time'] - current_segment['end_time']) < 2.0):
                current_segment['end_time'] = result['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                current_segment['confidence'] = (current_segment['confidence'] + result['confidence']) / 2
            else:
                # Save current and start new
                merged_results.append(current_segment)
                current_segment = result.copy()
        
        # Don't forget the last segment
        merged_results.append(current_segment)
    
    return merged_results, total_duration, paragraphs

# Option to choose analysis method
def analyze_podcast(audio_file_path, use_paragraphs=True):
    """Wrapper function to choose analysis method"""
    if use_paragraphs:
        results, total_duration, paragraphs = analyze_podcast_with_paragraphs(audio_file_path)
        return results, total_duration
    else:
        # Original 30-second chunk method
        return analyze_podcast_fixed_chunks(audio_file_path)

def analyze_podcast_fixed_chunks(audio_file_path, segment_duration=30):
    """Original fixed-chunk analysis method"""
    xgb_model, _ = load_models()
    if xgb_model is None:
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
        
        # Fix tempo processing if needed
        if 'tempo' in feature_df.columns:
            if feature_df['tempo'].dtype == 'object':
                feature_df['tempo'] = pd.to_numeric(feature_df['tempo'], errors='coerce').fillna(0)
        
        prediction_proba = xgb_model.predict_proba(feature_df)[0]
        
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

# ==========================================
# PODCAST SEARCH FEATURES
# ==========================================

def search_podcasts(query, limit=10):
    """Search for podcasts using iTunes API"""
    try:
        url = "https://itunes.apple.com/search"
        params = {
            'term': query,
            'media': 'podcast',
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        podcasts = []
        for item in data.get('results', []):
            podcasts.append({
                'name': item.get('collectionName', ''),
                'artist': item.get('artistName', ''),
                'feed_url': item.get('feedUrl', ''),
                'artwork': item.get('artworkUrl100', ''),
                'description': item.get('description', '')[:200] + '...' if item.get('description') else ''
            })
        
        return podcasts
    except Exception as e:
        st.error(f"Error searching podcasts: {e}")
        return []

def get_podcast_episodes(feed_url, limit=20):
    """Get recent episodes from a podcast RSS feed"""
    try:
        feed = feedparser.parse(feed_url)
        
        episodes = []
        for entry in feed.entries[:limit]:
            # Find the audio URL
            audio_url = None
            for link in entry.get('links', []):
                if link.get('type', '').startswith('audio/'):
                    audio_url = link.get('href')
                    break
            
            # Try enclosures if no direct audio link
            if not audio_url and hasattr(entry, 'enclosures'):
                for enclosure in entry.enclosures:
                    if enclosure.get('type', '').startswith('audio/'):
                        audio_url = enclosure.get('href')
                        break
            
            if audio_url:
                episodes.append({
                    'title': entry.get('title', 'Unknown Episode'),
                    'published': entry.get('published', ''),
                    'audio_url': audio_url,
                    'duration': entry.get('itunes_duration', 'Unknown'),
                    'description': entry.get('summary', '')[:200] + '...' if entry.get('summary') else ''
                })
        
        return episodes
    except Exception as e:
        st.error(f"Error fetching episodes: {e}")
        return []

def download_episode(audio_url, max_size_mb=50):
    """Download podcast episode to temporary file"""
    try:
        response = requests.get(audio_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                st.error(f"Episode too large ({size_mb:.1f}MB). Max size: {max_size_mb}MB")
                return None
        
        # Download to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        
        downloaded = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                downloaded += len(chunk)
                
                if downloaded > max_bytes:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    st.error(f"Episode too large (>{max_size_mb}MB)")
                    return None
        
        temp_file.close()
        return temp_file.name
        
    except Exception as e:
        st.error(f"Error downloading episode: {e}")
        return None

# ==========================================
# AD-FREE AUDIO PROCESSING
# ==========================================

@st.cache_data
def create_ad_free_audio(original_audio_path, results, output_format='mp3'):
    """Create ad-free version by removing ad segments - cached to prevent recreation"""
    try:
        audio = AudioSegment.from_file(original_audio_path)
        
        content_segments = []
        
        for result in results:
            if result['prediction'] == 'content':
                start_ms = int(result['start_time'] * 1000)
                end_ms = int(result['end_time'] * 1000)
                segment = audio[start_ms:end_ms]
                content_segments.append(segment)
        
        if content_segments:
            ad_free_audio = content_segments[0]
            for segment in content_segments[1:]:
                ad_free_audio += segment
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}')
            ad_free_audio.export(temp_file.name, format=output_format)
            
            # Read the file data
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return audio_data
        else:
            st.error("No content segments found!")
            return None
            
    except Exception as e:
        st.error(f"Error creating ad-free audio: {e}")
        return None

def format_duration(seconds):
    """Format duration in seconds to MM:SS or HH:MM:SS"""
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

# ==========================================
# ENHANCED RESULTS DISPLAY
# ==========================================

def display_enhanced_results(results, total_duration, original_audio_path):
    """Enhanced results display with all new features"""
    
    # Store results in session state so they persist
    st.session_state.analysis_results = results
    st.session_state.total_duration = total_duration
    st.session_state.original_audio_path = original_audio_path
    st.session_state.analysis_complete = True
    
    # Calculate stats
    ad_segments = [r for r in results if r['prediction'] == 'ad']
    ad_duration = sum([r['duration'] for r in ad_segments])
    content_duration = total_duration - ad_duration
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Duration", format_duration(total_duration))
    with col2:
        st.metric("Ad Segments", len(ad_segments))
    with col3:
        st.metric("Ad Duration", format_duration(ad_duration))
    with col4:
        time_saved_pct = (ad_duration / total_duration) * 100 if total_duration > 0 else 0
        st.metric("Time Saved", f"{format_duration(ad_duration)} ({time_saved_pct:.1f}%)")
    
    # Timeline visualization
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
    
    # Ad-Free Playback Section
    st.subheader("🎵 Ad-Free Podcast")
    
    # Format selection
    col1, col2 = st.columns([1, 3])
    with col1:
        output_format = st.selectbox("Choose format:", ["mp3", "wav"], index=0, key="format_select")
    
    # Create the ad-free audio immediately after analysis completes
    if 'ad_free_audio_data' not in st.session_state or st.session_state.get('last_format') != output_format:
        with st.spinner("Creating ad-free version..."):
            audio_data = create_ad_free_audio(original_audio_path, results, output_format)
            if audio_data:
                st.session_state.ad_free_audio_data = audio_data
                st.session_state.last_format = output_format
                st.session_state.content_duration = content_duration
    
    # Display download button if audio is ready
    if 'ad_free_audio_data' in st.session_state and st.session_state.ad_free_audio_data:
        audio_data = st.session_state.ad_free_audio_data
        file_size = len(audio_data) / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"✅ Ad-free file ready! Size: {file_size:.1f}MB | Duration: {format_duration(st.session_state.content_duration)}")
        
        with col2:
            mime_type = 'audio/mpeg' if output_format == 'mp3' else 'audio/wav'
            download_filename = f"podcast_ad_free.{output_format}"
            
            st.download_button(
                label="⬇️ Download Ad-Free Podcast",
                data=audio_data,
                file_name=download_filename,
                mime=mime_type,
                key="download_button_main"
            )
        
        # Audio player for browser playback
        if file_size < 25:  # Only show player for files under 25MB
            st.subheader("🎧 Play in Browser")
            b64_audio = base64.b64encode(audio_data).decode()
            audio_html = f"""
            <audio controls style="width: 100%;" preload="auto">
                <source src="data:{mime_type};base64,{b64_audio}" type="{mime_type}">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("File too large for browser playback. Please use the download button above.")
    
    # Detailed results
    with st.expander("📋 Detailed Results"):
        df = pd.DataFrame(results)
        df['start_min'] = df['start_time'] / 60
        df['end_min'] = df['end_time'] / 60
        df['confidence_pct'] = df['confidence'] * 100
        
        # Include text preview if available
        columns_to_show = ['start_min', 'end_min', 'prediction', 'confidence_pct']
        column_config = {
            'start_min': st.column_config.NumberColumn('Start (min)', format="%.1f"),
            'end_min': st.column_config.NumberColumn('End (min)', format="%.1f"),
            'prediction': st.column_config.TextColumn('Prediction'),
            'confidence_pct': st.column_config.NumberColumn('Confidence (%)', format="%.1f")
        }
        
        if 'text' in df.columns:
            columns_to_show.append('text')
            column_config['text'] = st.column_config.TextColumn('Text Preview')
        
        st.dataframe(
            df[columns_to_show],
            column_config=column_config
        )

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.title("🎧 Podcast Ad Detector")
    st.markdown("Upload your own file or search for podcasts to analyze for ads!")
    
    # Add analysis method selector in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        analysis_method = st.radio(
            "Analysis Method:",
            ["Paragraph-based (Recommended)", "Fixed 30-second chunks"],
            index=0,
            help="Paragraph-based analysis uses AI transcription to detect natural speech boundaries, providing more accurate results since the model was trained on paragraphs."
        )
        use_paragraphs = analysis_method == "Paragraph-based (Recommended)"
        
        if use_paragraphs:
            st.info("🎯 Using paragraph detection for better accuracy. First analysis may take longer due to transcription.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📁 Upload File", "🔍 Search Podcasts"])
    
    # Tab 1: File Upload
    with tab1:
        st.markdown("Upload your own podcast file:")
        uploaded_file = st.file_uploader(
            "Choose your podcast file",
            type=['mp3', 'wav', 'm4a'],
            help="Upload an audio file to analyze for advertisements"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Check if we need to analyze or just display existing results
            if st.button("🔍 Analyze for Ads", type="primary", key="upload_analyze"):
                with st.spinner("Processing your podcast... This may take a few minutes."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    results, total_duration = analyze_podcast(tmp_file_path, use_paragraphs=use_paragraphs)
                    
                    if results:
                        display_enhanced_results(results, total_duration, tmp_file_path)
            
            # If analysis was already done, show the results
            elif st.session_state.get('analysis_complete') and st.session_state.get('analysis_results'):
                display_enhanced_results(
                    st.session_state.analysis_results,
                    st.session_state.total_duration,
                    st.session_state.original_audio_path
                )