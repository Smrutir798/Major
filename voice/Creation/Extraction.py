import librosa
import numpy as np
import pandas as pd
import parselmouth
import antropy as ant

def extract_mdvp_features(audio_path):
    # Load audio safely
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            raise ValueError("Audio file is empty.")
    except Exception as e:
        raise RuntimeError(f"Error loading audio: {e}")

    # --- Fundamental Frequency (Fo, Fhi, Flo) ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    fo = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
    fhi = np.max(pitch_values) if len(pitch_values) > 0 else 0.0
    flo = np.min(pitch_values) if len(pitch_values) > 0 else 0.0

    # --- Praat Features (Jitter, Shimmer, HNR, RAP, PPQ, APQ, DDP, DDA) ---
    try:
        snd = parselmouth.Sound(audio_path)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60, 400)
        num_points = parselmouth.praat.call(point_process, "Get number of points")

        if num_points > 1:
            jitter_local = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_abs = parselmouth.praat.call([snd, point_process], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
            rap = parselmouth.praat.call([snd, point_process], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            ppq = parselmouth.praat.call([snd, point_process], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            ddp = 3 * rap

            shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            dda = 3 * apq3
        else:
            jitter_local = jitter_abs = rap = ppq = ddp = 0.0
            shimmer_local = shimmer_db = apq3 = apq5 = apq = dda = 0.0
    except Exception as e:
        jitter_local = jitter_abs = rap = ppq = ddp = 0.0
        shimmer_local = shimmer_db = apq3 = apq5 = apq = dda = 0.0

    # --- Harmonics-to-Noise Ratio (HNR) ---
    try:
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (1 + np.exp(hnr))  # Approximate inverse relation
    except:
        hnr = 0.0
        nhr = 0.0

    # --- Nonlinear Dynamics (RPDE, DFA, PPE) ---
    try:
        rpde = ant.num_potential(y)
    except:
        rpde = 0.0

    try:
        dfa = ant.detrended_fluctuation(y)
    except:
        dfa = 0.0

    try:
        ppe = np.std(librosa.feature.chroma_stft(y=y, sr=sr))
    except:
        ppe = 0.0

    # --- Spread1, Spread2, D2 ---
    try:
        spread1 = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spread2 = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        d2 = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    except:
        spread1 = spread2 = d2 = 0.0

    # Combine Features into DataFrame
    features = {
        "MDVP:Fo(Hz)": [fo],
        "MDVP:Fhi(Hz)": [fhi],
        "MDVP:Flo(Hz)": [flo],
        "NHR": [nhr],
        "HNR": [hnr],
        "DFA": [dfa],
        "D2": [d2],
        "PPE": [ppe]
    }

    return pd.DataFrame(features)


# Example Usage:
if __name__ == "__main__":
    audio_file = "VALUE OF TIME _ A Life Changing Motivational Story _ Time Story _ English Stories _ Moral Stories [9ofL45Mrzj0].wav"
    df = extract_mdvp_features(audio_file)
    print(df)
    # Save to CSV
    df.to_csv("extracted_features.csv", index=False)
    print("Features saved to extracted_features.csv")
