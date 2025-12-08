import streamlit as st
import pandas as pd
import pickle

# =====================================================
# 1. Load model pipeline (preprocess + KNN) dari .pkl
# =====================================================

@st.cache_resource
def load_model(path: str = "modelKNN_fixed-3.pkl"):
    """
    Load model dari file .pkl.
    - Di notebook kamu: disimpan sebagai dict {"model": pipeline, "feature_columns": [...]}
    """
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        # Biar kalau error kelihatan jelas di UI Streamlit
        st.error(
            f"Gagal load model dari '{path}'.\n\n"
            f"Jenis error: {type(e).__name__}\n"
            "Ini biasanya terjadi kalau:\n"
            "- Versi scikit-learn di Colab beda dengan di Streamlit, atau\n"
            "- Library yang dipakai saat training belum di-install di environment deploy."
        )
        raise

    if isinstance(obj, dict):
        model = obj.get("model", obj)
        feature_columns = obj.get("feature_columns", None)
    else:
        model = obj
        feature_columns = getattr(model, "feature_names_in_", None)

    return model, feature_columns


model, feature_cols = load_model()

# =====================================================
# 2. Fungsi: bangun 1 baris input mentah (SAMA seperti X di notebook)
# =====================================================

def build_input_row(
    gender: str,
    age: float,
    height: float,
    weight: float,
    fam_hist: str,
    favc: str,
    fcvc: float,
    ncp: float,
    caec: str,
    smoke: str,
    ch2o: float,
    scc: str,
    faf: float,
    tue: float,
    calc: str,
    mtrans: str,
) -> pd.DataFrame:
    """
    Struktur kolom HARUS sama seperti X di notebook:
    ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
     'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
     'CALC', 'MTRANS']
    """
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

    # Kalau kamu simpan feature_columns di .pkl, kita align urutan kolomnya
    if feature_cols is not None:
        df = df[feature_cols]

    return df


# =====================================================
# 3. Streamlit UI
# =====================================================

st.set_page_config(page_title="Obesity Classification (KNN)", layout="centered")

st.title("Klasifikasi Tingkat Obesitas")
st.caption("Model KNN + preprocessing (ColumnTransformer) dari notebook.")

st.success("Model pipeline berhasil di-load dari `modelKNN_fixed.pkl`.")

st.markdown("### Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

    age = st.number_input(
        "Age (tahun)",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
    )

    height = st.number_input(
        "Height (meter)",
        min_value=1.2,
        max_value=2.2,
        value=1.70,
        step=0.01,
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.5,
    )

    fam_hist = st.selectbox(
        "Family history with overweight",
        ["yes", "no"],
    )

    favc = st.selectbox(
        "Frequent consumption of high caloric food (FAVC)",
        ["yes", "no"],
    )

    smoke = st.selectbox(
        "Do you smoke? (SMOKE)",
        ["yes", "no"],
    )

    scc = st.selectbox(
        "Do you monitor the calories you eat daily? (SCC)",
        ["yes", "no"],
    )

with col2:
    fcvc = st.slider(
        "FCVC – Frequency of vegetables consumption (1–3)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.5,
    )

    ncp = st.slider(
        "NCP – Number of main meals per day (1–4)",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=1.0,
    )

    ch2o = st.slider(
        "CH2O – Water intake (1–3)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=1.0,
    )

    faf = st.slider(
        "FAF – Physical activity (0–4)",
        min_value=0.0,
        max_value=4.0,
        value=2.0,
        step=1.0,
    )

    tue = st.slider(
        "TUE – Time using technology devices (0–3)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=1.0,
    )

    caec = st.selectbox(
        "CAEC – Do you eat any food between meals?",
        ["no", "Sometimes", "Frequently", "Always"],
    )

    calc = st.selectbox(
        "CALC – How often do you drink alcohol?",
        ["no", "Sometimes", "Frequently", "Always"],
    )

    mtrans_label = st.selectbox(
        "MTRANS – Which transportation do you usually use?",
        ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"],
    )

# mapping label ke nilai di dataset
mtrans_map = {
    "Automobile": "Automobile",
    "Motorbike": "Motorbike",
    "Bike": "Bike",
    "Public Transportation": "Public_Transportation",
    "Walking": "Walking",
}
mtrans = mtrans_map[mtrans_label]

st.markdown("---")

if st.button("Prediksi Tingkat Obesitas"):
    # 1. Bentuk 1 baris DataFrame mentah (seperti X di notebook)
    row = build_input_row(
        gender=gender,
        age=age,
        height=height,
        weight=weight,
        fam_hist=fam_hist,
        favc=favc,
        fcvc=fcvc,
        ncp=ncp,
        caec=caec,
        smoke=smoke,
        ch2o=ch2o,
        scc=scc,
        faf=faf,
        tue=tue,
        calc=calc,
        mtrans=mtrans,
    )

    # 2. Prediksi pakai pipeline (preprocess + KNN)
    y_pred = model.predict(row)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"**Kelas NObeyesdad (prediksi):** `{y_pred}`")

    if proba is not None:
        proba_df = pd.DataFrame(
            {"Class": model.classes_, "Probability": proba}
        )
        st.dataframe(proba_df, use_container_width=True)

    st.info(
        "Model ini digunakan untuk kebutuhan edukasi dan eksperimen. "
        "Untuk keputusan medis/klinis, tetap konsultasikan dengan tenaga kesehatan profesional."
    )
