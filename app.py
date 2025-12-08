import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================================================
# 1. Load model dari file PKL (TANPA TRAIN ULANG)
# ======================================================

@st.cache_resource
def load_model(path: str = "modelKNN_fixed.pkl"):
    """
    Load model dari .pkl.
    Diasumsikan .pkl berisi:
      - dict {"model": <knn>, "feature_columns": [...]}  ATAU
      - langsung objek model sklearn
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Kalau isinya dict: ambil model & feature_columns
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        feature_columns = obj.get("feature_columns", None)
    else:
        model = obj
        feature_columns = None

    # Coba ambil feature_names_in_ kalau feature_columns tidak disimpan eksplisit
    if feature_columns is None and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    if feature_columns is None:
        raise ValueError(
            "Tidak menemukan 'feature_columns' di dalam PKL dan model tidak punya "
            "atribut 'feature_names_in_'. "
            "Saat menyimpan model, sebaiknya simpan juga daftar kolom fitur."
        )

    return model, list(feature_columns)


model, feature_cols = load_model()


# ======================================================
# 2. PREPROCESSING INPUT (SESUAI SCRIPT PY KAMU)
#    - hitung BMI
#    - binning AgeGroup, BMIGroup
#    - get_dummies (one hot) -> lalu align ke feature_cols
# ======================================================

NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Kolom kategorik berasal dari dataset Obesity:
CATEGORICAL_BASE = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]


def add_bmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah BMI, AgeGroup, BMIGroup seperti di script .py"""
    df = df.copy()

    # BMI
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # AgeGroup
    age_bins = [0, 18, 30, 50, 100]
    age_labels = ["Teen", "YoungAdult", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    # BMIGroup (WHO-ish)
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMIGroup"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)

    return df


def encode_for_model(df_raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Replikasi bagian:
        df_encoded = pd.get_dummies(df_clean,
             columns=categorical_features + ['AgeGroup','BMIGroup'],
             drop_first=True)

    Tapi di sini kita hanya punya 1 baris data (input user).
    Setelah itu, kita reindex ke feature_columns hasil training.
    """
    df = add_bmi_features(df_raw)

    # sama konsepnya dengan script:
    numeric_features = NUMERIC_COLS
    categorical_features = CATEGORICAL_BASE

    df_enc = pd.get_dummies(
        df,
        columns=categorical_features + ["AgeGroup", "BMIGroup"],
        drop_first=True,
    )

    # Buang target kalau ada
    df_enc = df_enc.drop(columns=["NObeyesdad"], errors="ignore")

    # Align ke urutan & set kolom yang sama dengan data training
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)

    return df_enc


def build_single_input_row(
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
    Bentuk 1 baris DataFrame mentah dengan kolom sama seperti dataset asli.
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
        "NObeyesdad": [np.nan],  # dummy target
    }
    return pd.DataFrame(data)


# ======================================================
# 3. STREAMLIT UI
# ======================================================

st.set_page_config(page_title="Obesity Level Prediction (KNN - PKL)", layout="centered")
st.title("Prediksi Tingkat Obesitas")
st.caption("Menggunakan model KNN yang sudah di-train dan disimpan di `modelKNN_fixed.pkl`")

st.success("Model berhasil di-load dari `modelKNN_fixed.pkl` (tanpa training ulang).")

st.markdown("### Isi Data Input")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age (tahun)", min_value=10.0, max_value=100.0, value=25.0, step=1.0)
    height = st.number_input("Height (meter)", min_value=1.2, max_value=2.2, value=1.70, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

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
        "NCP – Number of main meals per day",
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
        [
            "Automobile",
            "Motorbike",
            "Bike",
            "Public Transportation",
            "Walking",
        ],
    )

# mapping label ke nilai asli dataset
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
    # 1. Bangun 1 baris data mentah
    raw_row = build_single_input_row(
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

    # 2. Preprocess (BMI, AgeGroup, BMIGroup, get_dummies, reindex)
    X_new = encode_for_model(raw_row, feature_cols)

    # 3. Prediksi pakai model dari PKL
    y_pred = model.predict(X_new)[0]

    try:
        proba = model.predict_proba(X_new)[0]
    except Exception:
        proba = None

    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi kelas NObeyesdad:** `{y_pred}`")

    if proba is not None:
        proba_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": proba
        })
        st.dataframe(proba_df, use_container_width=True)

    st.info(
        "Model ini untuk tujuan edukasi. Untuk keputusan medis/klinis, "
        "tetap konsultasikan dengan tenaga kesehatan profesional."
    )
