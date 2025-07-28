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

# Page config 
st.set_page_config(
    page_title="🎧 Podcast Ad Detector",
    page_icon="🎧",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('./feature_analysis/models/xgb_best_model_20250727.joblib')

def extract_segment_features(audio_file, start_time, duration):
    """Your existing feature extraction function - unchanged"""
    
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

def analyze_podcast(audio_file_path, segment_duration=30):
    """Your existing analyze function with tempo fix"""
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
        
        # Fix tempo processing if needed
        if 'tempo' in feature_df.columns:
            if feature_df['tempo'].dtype == 'object':
                feature_df['tempo'] = pd.to_numeric(feature_df['tempo'], errors='coerce').fillna(0)
        
        prediction_proba = model.predict_proba(feature_df)[0]
        
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
# NEW FEATURES: Podcast Search
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
# NEW FEATURES: Ad-Free Playback
# ==========================================

def create_ad_free_audio(original_audio_path, results, output_format='mp3'):
    """Create ad-free version by removing ad segments"""
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
            
            return temp_file.name
        else:
            st.error("No content segments found!")
            return None
            
    except Exception as e:
        st.error(f"Error creating ad-free audio: {e}")
        return None

def get_audio_player_html(file_path):
    """Create HTML audio player for browser playback"""
    try:
        # Check file size first
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        if file_size > 25:  # Browser limit for data URLs
            st.warning(f"File too large for browser playback ({file_size:.1f}MB). Try downloading instead.")
            return None
        
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        b64_audio = base64.b64encode(audio_bytes).decode()
        mime_type = 'audio/mpeg' if file_path.endswith('.mp3') else 'audio/wav'
        
        # Debug info
        st.write(f"**Debug:** File size: {file_size:.1f}MB, MIME: {mime_type}")
        st.write(f"**Debug:** Base64 length: {len(b64_audio)} characters")
        
        audio_html = f"""
        <div style="margin: 20px 0;">
            <audio controls style="width: 100%;" preload="auto">
                <source src="data:{mime_type};base64,{b64_audio}" type="{mime_type}">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
        return audio_html
        
    except Exception as e:
        st.error(f"Error creating audio player: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def get_audio_download_link(file_path, filename):
    """Generate download link for audio file"""
    try:
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        if file_size > 25:  # Browser limit for data URLs
            st.warning(f"File too large for direct download ({file_size:.1f}MB). File saved locally.")
            st.info(f"File location: {file_path}")
            return None
        
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        b64_audio = base64.b64encode(audio_bytes).decode()
        mime_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/wav'
        
        # Debug info
        st.write(f"**Download Debug:** File size: {file_size:.1f}MB")
        
        download_link = f"""
        <a href="data:{mime_type};base64,{b64_audio}" download="{filename}" 
           style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; 
                  color: white; text-decoration: none; border-radius: 5px; margin: 10px 0;">
            ⬇️ Download Ad-Free Podcast ({file_size:.1f}MB)
        </a>
        """
        return download_link
        
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

# Alternative: Use Streamlit's built-in download button
def create_streamlit_download_button(file_path, filename):
    """Use Streamlit's native download button"""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        mime_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/wav'
        
        st.download_button(
            label="⬇️ Download Ad-Free Podcast",
            data=audio_bytes,
            file_name=filename,
            mime=mime_type
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error creating download button: {e}")
        return False

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
    
    # NEW: Ad-Free Playback Section
    st.subheader("🎵 Ad-Free Podcast")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎧 Play in Browser", type="primary"):
            st.session_state.create_player = True
    
    with col2:
        if st.button("⬇️ Download File"):
            st.session_state.create_download = True
    
    with col3:
        output_format = st.selectbox("Format:", ["mp3", "wav"], index=0)
    
    # Handle playback/download
    if st.session_state.get('create_player') or st.session_state.get('create_download'):
        with st.spinner("Creating ad-free version..."):
            ad_free_path = create_ad_free_audio(original_audio_path, results, output_format)
        
        if ad_free_path:
            # Show file info
            file_size = os.path.getsize(ad_free_path) / (1024 * 1024)
            st.info(f"✅ Ad-free file created! Size: {file_size:.1f}MB | Duration: {format_duration(content_duration)}")
            
            if st.session_state.get('create_player'):
                st.subheader("🎧 Play Ad-Free Podcast")
                
                if file_size < 25:  # Small enough for browser
                    audio_player = get_audio_player_html(ad_free_path)
                    if audio_player:
                        st.markdown(audio_player, unsafe_allow_html=True)
                        st.success("✅ Playing in browser!")
                    else:
                        st.error("❌ Could not create audio player")
                else:
                    st.warning("File too large for browser playback. Please download instead.")
                
                st.session_state.create_player = False
            
            if st.session_state.get('create_download'):
                st.subheader("⬇️ Download Ad-Free Podcast")
                
                download_filename = f"podcast_ad_free.{output_format}"
                
                # Try Streamlit's native download button first
                if create_streamlit_download_button(ad_free_path, download_filename):
                    st.success("✅ Click the download button above!")
                else:
                    # Fallback to HTML download link
                    download_link = get_audio_download_link(ad_free_path, download_filename)
                    if download_link:
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not create download link")
                        st.info(f"File saved at: {ad_free_path}")
                
                st.session_state.create_download = False
        else:
            st.error("❌ Failed to create ad-free audio")
    
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

# ==========================================
# MAIN APP WITH TABS
# ==========================================

def main():
    st.title("🎧 Podcast Ad Detector")
    st.markdown("Upload your own file or search for podcasts to analyze for ads!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📁 Upload File", "🔍 Search Podcasts"])
    
    # Tab 1: File Upload (your existing functionality)
    with tab1:
        st.markdown("Upload your own podcast file:")
        uploaded_file = st.file_uploader(
            "Choose your podcast file",
            type=['mp3', 'wav', 'm4a'],
            help="Upload an audio file to analyze for advertisements"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("🔍 Analyze for Ads", type="primary", key="upload_analyze"):
                with st.spinner("Processing your podcast... This may take a few minutes."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    results, total_duration = analyze_podcast(tmp_file_path)
                    
                    if results:
                        display_enhanced_results(results, total_duration, tmp_file_path)
    
    # Tab 2: Podcast Search (new functionality)
    with tab2:
        st.markdown("Search and select podcasts from iTunes:")
        
        search_query = st.text_input("Search for podcasts:", placeholder="e.g., 'Joe Rogan', 'NPR News', 'Comedy'")
        
        if search_query:
            with st.spinner("Searching podcasts..."):
                podcasts = search_podcasts(search_query)
            
            if podcasts:
                st.subheader(f"Found {len(podcasts)} podcasts:")
                
                for i, podcast in enumerate(podcasts):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            if podcast['artwork']:
                                st.image(podcast['artwork'], width=100)
                        
                        with col2:
                            st.write(f"**{podcast['name']}**")
                            st.write(f"By: {podcast['artist']}")
                            if podcast['description']:
                                st.write(podcast['description'])
                            
                            if st.button(f"View Episodes", key=f"select_{i}"):
                                st.session_state.selected_podcast = podcast
                                st.session_state.show_episodes = True
                        
                        st.divider()
        
        # Show episodes if podcast selected
        if st.session_state.get('show_episodes') and st.session_state.get('selected_podcast'):
            podcast = st.session_state.selected_podcast
            
            st.subheader(f"Recent Episodes: {podcast['name']}")
            
            with st.spinner("Loading episodes..."):
                episodes = get_podcast_episodes(podcast['feed_url'])
            
            if episodes:
                for i, episode in enumerate(episodes):
                    with st.expander(f"🎧 {episode['title']}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Published:** {episode['published']}")
                            st.write(f"**Duration:** {episode['duration']}")
                            if episode['description']:
                                st.write(episode['description'])
                        
                        with col2:
                            if st.button("Analyze Episode", key=f"analyze_{i}"):
                                with st.spinner("Downloading and analyzing episode..."):
                                    temp_file_path = download_episode(episode['audio_url'])
                                    
                                    if temp_file_path:
                                        results, total_duration = analyze_podcast(temp_file_path)
                                        
                                        if results:
                                            st.success(f"✅ Analysis complete for: {episode['title']}")
                                            display_enhanced_results(results, total_duration, temp_file_path)
                                        
                                        # Clean up
                                        os.unlink(temp_file_path)

if __name__ == "__main__":
    # Initialize session state
    if 'show_episodes' not in st.session_state:
        st.session_state.show_episodes = False
    
    main()