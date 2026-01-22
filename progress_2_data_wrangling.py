# ====================================================================
# PROJECT PROGRESS 2/3: DATA IMPORT, WRANGLING & FEATURE EXTRACTION
# Cat Sound Classification - Bioinformatics Project
# ====================================================================

import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. IMPORTING DATASET
# ====================================================================
print("="*70)
print("STEP 1: IMPORTING DATASET")
print("="*70)

DATA_DIR = r'C:\PROGRAMMING FOR BIOINFO\Cat_Meow_Classification\dataset\dataset'

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory {DATA_DIR} not found!")

audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
print(f"✓ Found {len(audio_files)} audio files")
print(f"✓ Sample files: {audio_files[:3]}")

# ====================================================================
# 2. PARSING METADATA FROM FILENAMES
# ====================================================================
def parse_filename(filename):
    """Extract metadata from filename"""
    parts = filename.replace('.wav', '').split('_')
    return {
        'filename': filename,
        'context': parts[0],  # B/F/I
        'cat_id': parts[1],
        'breed': parts[2],    # MC/EU
        'sex': parts[3],      # FI/FN/MI/MN
        'owner_id': parts[4],
        'session': parts[5][0],  
        'vocalization_num': parts[5][1:]
    }

# Label mappings
context_map = {'F': 'Waiting for Food', 'I': 'Isolation', 'B': 'Brushing'}
breed_map = {'MC': 'Maine Coon', 'EU': 'European Shorthair'}
sex_map = {'FI': 'Female Intact', 'FN': 'Female Neutered', 'MI': 'Male Intact', 'MN': 'Male Neutered'}

# ====================================================================
# 3. AUDIO CLEANSING FUNCTION
# ====================================================================
def cleanse_audio(file_path, target_sr=16000):
    """Trim, resample, and normalize audio"""
    data, sr = librosa.load(file_path, sr=None)
    data, _ = librosa.effects.trim(data, top_db=20)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    data = librosa.util.normalize(data)
    return data, sr

# ====================================================================
# 4. FEATURE EXTRACTION FUNCTION
# ====================================================================
def extract_features(file_path, target_sr=16000):
    """Extract 32 audio features: MFCC (13) + Chroma (12) + Spectral Contrast (7)"""
    data, sr = cleanse_audio(file_path, target_sr)
    
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=data, sr=sr)
    
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1)
    ])
    return features

# ====================================================================
# 5. PROCESS ALL AUDIO FILES
# ====================================================================
dataset = []

print(f"\n⚙️  Processing {len(audio_files)} files...")

for i, file in enumerate(audio_files):
    if (i + 1) % 10 == 0:
        print(f"   Processed {i+1}/{len(audio_files)} files...")
    
    file_path = os.path.join(DATA_DIR, file)
    
    try:
        features = extract_features(file_path)
        metadata = parse_filename(file)
        
        record = {
            'filename': file,
            'features': features,
            'context': metadata['context'],
            'context_label': context_map[metadata['context']],
            'breed': metadata['breed'],
            'breed_label': breed_map[metadata['breed']],
            'sex': metadata['sex'],
            'cat_id': metadata['cat_id'],
            'owner_id': metadata['owner_id']
        }
        dataset.append(record)
        
    except Exception as e:
        print(f"   ✗ Error processing {file}: {e}")

print(f"\n✓ Successfully processed {len(dataset)} audio files")

# ====================================================================
# 6. CREATE DATAFRAME AND ONE-HOT ENCODING
# ====================================================================
df = pd.DataFrame(dataset)
df = pd.get_dummies(df, columns=['context', 'breed', 'sex'], prefix=['ctx', 'breed', 'sex'])

print(f"\n📊 Final Dataset Shape: {df.shape}")
print(df[['filename', 'context_label', 'breed_label']].head())

# ====================================================================
# 7. SAVE FEATURES AND METADATA
# ====================================================================
features_array = np.array(df['features'].tolist())
np.save('cat_features.npy', features_array)

metadata_cols = [col for col in df.columns if col != 'features']
df[metadata_cols].to_csv('cat_metadata.csv', index=False)

print("\n✓ Exported files:")
print("   • cat_features.npy")
print("   • cat_metadata.csv")

print("\nDATA WRANGLING COMPLETE!")
