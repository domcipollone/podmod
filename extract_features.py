import os
import pandas as pd
import numpy as np
import librosa
import warnings
from tqdm import tqdm
import traceback 
import multiprocessing as mp 
from functools import partial 
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')


class QueryDatabase: 

    def __init__(self):
        self.db_engine = create_engine(os.getenv("DB_CONN_STRING"))
        print("Built DB engine")

    def write_data(self, df, table_name): 

        try: 
            print(f"Attempting to write {len(df)} rows to {table_name}")

            with self.db_engine.connect() as conn: 
                df.to_sql(name=table_name, con=conn, if_exists='append', index=False, chunksize=1000, method='multi')

            print(f"Sucessfully wrote data to {table_name}")
            print("Disposing of Engine")

            self.db_engine.dispose()

            return self

        except Exception: 
            self.db_engine.dispose()
            print(f"Failed to write data to {table_name}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None
        
    def read_data(self, query):

        try: 
            print("Attempting to read data")
            with self.db_engine.connect() as conn: 
                df = pd.read_sql_query(sql=query, con=conn)

            print("Sucessfully read data")
            print("Disposing of Engine")

            return df
        
        except Exception as e:
            self.db_engine.dispose()
            print(f"Failed to read data: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None 


def extract_segment_features(audio_file, start_time, duration):
    """
    Extract comprehensive features from a single audio segment
    
    Returns:
        Dictionary with all extracted features
    """
    
    # Load audio segment
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration, sr=22050)
    
    if len(y) == 0:
        # raise ValueError(f"Empty audio segment")
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
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # 4. Spectral Contrast - difference between peaks and valleys in spectrum
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # 5. Spectral Flatness - measure of how "noise-like" vs "tone-like" the sound is
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # ==========================================
    # TEMPORAL FEATURES (Time Domain)
    # ==========================================
    
    # 6. RMS Energy - overall loudness/energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 7. Zero Crossing Rate - how often signal crosses zero (roughness indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # 8. Basic statistical features
    features['amplitude_mean'] = np.mean(np.abs(y))
    features['amplitude_std'] = np.std(y)
    features['amplitude_max'] = np.max(np.abs(y))
    features['amplitude_min'] = np.min(y)
    features['dynamic_range'] = np.max(y) - np.min(y)
    
    # ==========================================
    # TIMBRAL FEATURES (MFCC - Speech Characteristics)
    # ==========================================
    
    # 9. MFCCs - Mel-frequency cepstral coefficients (speech/music characteristics)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
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
        features['chroma_var'] = np.var(chroma)
    except:
        features['chroma_mean'] = 0
        features['chroma_std'] = 0
        features['chroma_var'] = 0
    
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
    
    # 14. Spectral rolloff percentiles (different cutoff points)
    for percent in [0.85, 0.95]:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=percent)[0]
        features[f'spectral_rolloff_{int(percent*100)}_mean'] = np.mean(rolloff)
    
    # 15. Polyfeatures - multiple spectral statistics
    poly_features = librosa.feature.poly_features(y=y, sr=sr, order=1)[0]
    features['poly_features_mean'] = np.mean(poly_features)
    features['poly_features_std'] = np.std(poly_features)
    
    return features

def process_row(row, training_path): 
    audio_file = row['audio_file']
    full_path = os.path.join(training_path, audio_file)

        # handle audio file not existing 
    if os.path.isfile(full_path):

        features = extract_segment_features(audio_file=full_path,
                                            start_time=row['start_time'],
                                            duration=row['duration']
        )
        
        if features is not None: 
            features.update({
                    'audio_file': row['audio_file'],
                    'transcript_id': row['transcript_id'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'duration': row['duration'],
                    'confidence': row['confidence'],
                    'target': row['label'],
                })
                
            return features # returns dictionary of features 
        
        else: 
            return None 
    
    else: 
        return None 

if __name__ == "__main__":

    q = QueryDatabase()

    segment_query = f"""
        SELECT 
        tl.audio_file,
        pod.transcript_id,
        pod.start_time::NUMERIC / 1000 AS start_time,
        pod.end_time::NUMERIC / 1000 AS end_time,
        (pod.end_time - pod.start_time)::NUMERIC / 1000 AS duration, 
        pod.confidence, pod.label
        FROM podcast_segment_labels pod  
        INNER JOIN transcript_log tl 
        USING (transcript_id) 
        ORDER BY pod.transcript_id, 3 
    """

    df = q.read_data(query=segment_query)
    training_path = "training_data/audio/"

    feature_list = []

    rows = df.to_dict('records')

    process_func = partial(process_row, training_path=training_path)

    # Use multiprocessing
    num_cores = mp.cpu_count() - 1  # Leave one core free
    print(f"Using {num_cores} cores to process {len(rows)} segments")
    
    with mp.Pool(processes=num_cores) as pool:
        # Use imap for progress tracking with tqdm
        feature_list = list(tqdm(
            pool.imap(process_func, rows), 
            total=len(rows),
            desc="Processing segments"
        ))

    feature_list = [f for f in feature_list if f is not None]        
    features_df = pd.DataFrame(feature_list)

    features_df.to_csv('feature_analysis/features_20250725.csv', index=False)
