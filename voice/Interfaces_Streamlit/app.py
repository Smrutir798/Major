import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io

# ---------------------
# Load Pretrained Model
# ---------------------
@st.cache_resource
def load_model():
    with open("rf_model_parkinson.pkl", "rb") as f:   # change filename to your actual model name
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="centered")
st.markdown("""
<style>
    .main {background-color: #f7f7fa;}
    .stButton>button {background-color: #4F8BF9; color: white; font-weight: bold;}
    .stDownloadButton>button {background-color: #4F8BF9; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Parkinson's Disease Prediction By Voice Data")
st.markdown(
    """
    <div style='color:#4F8BF9;font-size:18px;'>
    Upload a CSV file or enter feature values manually to predict Parkinson's Disease.<br>
    You can also upload a WAV file to visualize the sound wave.
    </div>
    """,
    unsafe_allow_html=True
)

# Define the features (adjust to match your training features order!)
feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "NHR", "HNR", "DFA", "D2", "PPE"
]

# Add Streamlit tabs for main app and sound analysis
st.markdown("# Parkinson's Disease Prediction App")
tabs = st.tabs(["Prediction & Data", "Sound Analysis"])

with tabs[0]:
    # ---------------------
    # Option 1: Manual Entry
    # ---------------------
    st.markdown("---")
    st.subheader("1️⃣ Enter Feature Values Manually")
    col1, col2, col3, col4 = st.columns(4)
    inputs = []
    for i, feature in enumerate(feature_names):
        with [col1, col2, col3, col4][i % 4]:
            value = st.number_input(f"{feature}", value=0.0, key=feature)
            inputs.append(value)

    if st.button("Predict from Manual Input"):
        X_input = np.array(inputs).reshape(1, -1)
        prediction = model.predict(X_input)[0]
        st.markdown("### Feature Values")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(feature_names, inputs, color="#4F8BF9")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Value')
        plt.tight_layout()
        st.pyplot(fig)
        if prediction == 1:
            st.success("✅ Model predicts: Healthy")
            st.toast("Healthy", icon="✅")
        else:
            st.error("⚠️ Model predicts: Parkinson's Disease")
            st.toast("Parkinson's Disease", icon="⚠️")

    # ---------------------
    # Option 2: CSV Upload
    # ---------------------
    st.markdown("---")
    st.subheader("2️⃣ Or Upload a CSV File")
    uploaded_file = st.file_uploader("Upload CSV with same feature columns", type=["csv"], key="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        # Predict
        preds = model.predict(df)
        df["Prediction"] = ["Healthy" if p == 1 else "Parkinson" for p in preds]
        st.write("Predictions:")
        st.dataframe(df)
        # Show popup for each prediction
        for i, p in enumerate(preds):
            if p == 1:
                st.toast(f"Sample {i+1}: Healthy", icon="✅")
            else:
                st.toast(f"Sample {i+1}: Parkinson's Disease", icon="⚠️")
    
        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
        # Feature distribution plot
        st.markdown("#### Feature Distribution (First Row)")
        if not df.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.bar(feature_names, df.iloc[0][feature_names], color="#F9A14F")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Value')
            plt.tight_layout()
            st.pyplot(fig2)

with tabs[1]:
    st.header("🔊 Sound Analysis Section")
    st.markdown("This section allows you to upload a WAV file, visualize its audio properties, and extract features for Parkinson's analysis.")
    st.markdown("---")
    st.subheader("1️⃣ Upload and Visualize Your Voice (WAV File)")

    def extract_mdvp_features(audio_path):
        import parselmouth
        import antropy as ant
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if len(y) == 0:
                raise ValueError("Audio file is empty.")
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        fo = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        fhi = np.max(pitch_values) if len(pitch_values) > 0 else 0.0
        flo = np.min(pitch_values) if len(pitch_values) > 0 else 0.0
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
        try:
            harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            nhr = 1 / (1 + np.exp(hnr))
        except:
            hnr = 0.0
            nhr = 0.0
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
        try:
            spread1 = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spread2 = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            d2 = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        except:
            spread1 = spread2 = d2 = 0.0
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

    audio_file = st.file_uploader("Upload a WAV file to see the sound wave", type=["wav"], key="wav")
    if audio_file is not None:
        y, sr = librosa.load(audio_file, sr=None)
        st.audio(audio_file, format='audio/wav')
        st.markdown("#### Sound Wave:")
        fig3, ax3 = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax3, color="#4F8BF9")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown("---")
        st.markdown("#### Spectrogram:")
        fig4, ax4 = plt.subplots(figsize=(8, 2))
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax4, cmap='magma')
        plt.colorbar(img, ax=ax4, format="%+2.0f dB")
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown("---")
        st.markdown("#### MFCCs:")
        fig5, ax5 = plt.subplots(figsize=(8, 2))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, x_axis='time', ax=ax5, sr=sr, cmap='coolwarm')
        plt.colorbar(img2, ax=ax5)
        plt.ylabel('MFCC')
        plt.tight_layout()
        st.pyplot(fig5)
        st.markdown("---")
        st.subheader("2️⃣ Extracted Features from Audio")
        with open("temp_uploaded.wav", "wb") as f:
            f.write(audio_file.read())
        try:
            features_df = extract_mdvp_features("temp_uploaded.wav")
            st.markdown("#### Extracted Features Table:")
            st.dataframe(features_df)
            csv_feat = features_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Extracted Features as CSV", csv_feat, "extracted_features.csv", "text/csv")
        except Exception as e:
            st.error(f"Feature extraction failed: {e}")
    st.markdown("---")
