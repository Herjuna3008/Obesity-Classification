import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai

# =====================================================
# 1. Load model pipeline (preprocess + KNN) dari .pkl
# =====================================================

@st.cache_resource
def load_model(path: str = "modelKNN_fixed-3.pkl"):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        st.error(
            f"Gagal load model dari '{path}'.\n\n"
            f"Jenis error: {type(e).__name__}\n"
            "Pastikan file .pkl ada di folder yang sama dan nama filenya benar."
        )
        return None, None

    # Handle jika yang disimpan adalah dict atau object langsung
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        feature_columns = obj.get("feature_columns", None)
    else:
        model = obj
        feature_columns = getattr(model, "feature_names_in_", None)

    return model, feature_columns

# Load Model
model, feature_cols = load_model()

# =====================================================
# 2. Fungsi: bangun 1 baris input mentah
# =====================================================

def build_input_row(gender, age, height, weight, fam_hist, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans):
    data = {
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [fam_hist],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "SCC": [scc],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans],
    }
    df = pd.DataFrame(data)
    # align urutan kolomnya jika ada info feature columns
    if feature_cols is not None:
        df = df[feature_cols]
    return df

# =============================
# 3. Fungsi Gen AI (Gemini)
# =============================

def get_gemini_advice(api_key, prediction, user_data, bmi):
    """Mengirim data ke Gemini untuk dianalisa"""
    try:
        genai.configure(api_key=api_key)
        model_ai = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prompt yang dikirim ke AI
        prompt = f"""
        Kamu adalah seorang Dokter Spesialis Gizi dan Kesehatan.
        
        Analisa profil pasien berikut:
        - **Diagnosa AI (KNN):** {prediction}
        - **BMI:** {bmi:.2f}
        - **Usia:** {user_data['Age']} tahun, {user_data['Gender']}
        - **Kebiasaan:**
            - Aktivitas Fisik: {user_data['FAF']} (Skala 0-3)
            - Makan Sayur: {user_data['FCVC']} (Skala 1-3)
            - Minum Air: {user_data['CH2O']} Liter
            - Merokok: {user_data['SMOKE']}
            - Riwayat Keluarga: {user_data['family_history_with_overweight']}
        
        Tugasmu:
        1. Jelaskan secara singkat **RISIKO KESEHATAN** utama jika kondisi ini berlanjut (maksimal 2 poin).
        2. Berikan 3 **REKOMENDASI MEDIS** yang spesifik, ramah, dan bisa langsung dipraktekkan berdasarkan data kebiasaan di atas.
        
        Gunakan Bahasa Indonesia yang profesional namun mudah dimengerti.
        """
        
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghubungi Gemini: {e}"

# =============================
# 4. Streamlit UI
# =============================

st.set_page_config(page_title="Obesity Classification (KNN)", layout="centered")

api_key = st.secrets["API_KEY"]

print("Mencari model yang tersedia...")
try:
    for m in genai.list_models():
        # Kita hanya cari model yang bisa generate text (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
except Exception as e:
    print(f"Error: {e}")

st.title("Klasifikasi Tingkat Obesitas")
st.caption("Menggunakan Model KNN")

if model:
    st.success("Model berhasil di-load.")
else:
    st.stop() # Berhenti jika model gagal load

st.markdown("### Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age (tahun)", 10.0, 100.0, 25.0, 1.0)
    height = st.number_input("Height (meter)", 1.2, 2.2, 1.70, 0.01)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
    fam_hist = st.selectbox("Family history with overweight", ["yes", "no"])
    favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
    smoke = st.selectbox("Do you smoke? (SMOKE)", ["yes", "no"])
    scc = st.selectbox("Do you monitor the calories you eat daily? (SCC)", ["yes", "no"])

with col2:
    fcvc = st.slider("FCVC – Frequency of vegetables consumption", 1.0, 3.0, 2.0, 0.5)
    ncp = st.slider("NCP – Number of main meals per day", 1.0, 4.0, 3.0, 1.0)
    ch2o = st.slider("CH2O – Water intake (Liters)", 1.0, 3.0, 2.0, 1.0)
    faf = st.slider("FAF – Physical activity frequency", 0.0, 3.0, 2.0, 1.0)
    tue = st.slider("TUE – Time using technology devices", 0.0, 2.0, 1.0, 1.0)
    caec = st.selectbox("CAEC – Eat food between meals?", ["no", "Sometimes", "Frequently", "Always"])
    calc = st.selectbox("CALC – Alcohol consumption?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans_label = st.selectbox("MTRANS – Transportation?", ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])

# Mapping Transportation
mtrans_map = {
    "Automobile": "Automobile", "Motorbike": "Motorbike", "Bike": "Bike",
    "Public Transportation": "Public_Transportation", "Walking": "Walking"
}
mtrans = mtrans_map[mtrans_label]

st.markdown("---")

if st.button("Prediksi Tingkat Obesitas"):
    # 1. Bentuk Data Input
    row = build_input_row(gender, age, height, weight, fam_hist, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans)

    # 2. Prediksi Model KNN
    y_pred = model.predict(row)[0]
    
    # Cek Probabilitas (Opsional)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]

    # 3. Tampilkan Hasil Prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"**Anda masuk dalam kategori:** `{y_pred}`")

    if proba is not None:
        with st.expander("Lihat Detail Probabilitas"):
            proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
            st.dataframe(proba_df, use_container_width=True)

    # 4. Integrasi Gemini AI (Perbaikan Indentasi & Variabel)
    st.markdown("---")
    st.subheader("Analisa Singkat dari Gemini AI")

    if not api_key:
        st.warning("⚠️ Error API_KEY ga kebaca")
    else:
        with st.spinner("Gemini AI sedang menganalisa data medis Anda..."):
            # Hitung BMI untuk info ke AI
            bmi_val = weight / (height ** 2)
            user_data_ai = {
                'Age': age,
                'Gender': gender,           
                'FAF': faf,
                'FCVC': fcvc,
                'CH2O': ch2o,
                'SMOKE': smoke,             
                'family_history_with_overweight': fam_hist
            }
            
            # Panggil fungsi Gemini (Gunakan y_pred sebagai hasil prediksi)
            analysis = get_gemini_advice(api_key, y_pred, user_data_ai, bmi_val)
            
            # Tampilkan Output
            st.markdown(analysis)
st.info(
    "Catatan: Model ini hanya untuk edukasi. Analisa AI dapat mungkin dapat melakukan kesalahan. "
    "Konsultasikan dengan dokter untuk diagnosa medis."
)
