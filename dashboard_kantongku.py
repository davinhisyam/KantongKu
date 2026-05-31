import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

st.set_page_config(page_title="KantongKu Dashboard V3", page_icon="👛", layout="wide")

st.title("👛 KantongKu – AI Receipt Analytics Dashboard")
st.caption("Capstone Dashboard untuk EDA, Data Quality, dan Kesiapan Model AI")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "category" in df.columns:
        df["category"] = df["category"].replace({
            "Makan dan Minuman": "Makanan & Minuman"
        })

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

uploaded = st.sidebar.file_uploader("Upload Final Dataset", type=["csv"])

if not uploaded:
    st.info("Upload dataset final untuk menampilkan dashboard.")
    st.stop()

df = load_data(uploaded)

# KPI
st.header("📌 Ringkasan Dataset")

c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric("Total Transaksi", len(df))
c2.metric("Merchant Unik", df["merchant"].nunique() if "merchant" in df.columns else "-")
c3.metric("Kategori Unik", df["category"].nunique() if "category" in df.columns else "-")
c4.metric("Missing Value", int(df.isna().sum().sum()))
c5.metric("Duplicate", int(df.duplicated().sum()))
c6.metric("Transaksi Negatif",
          int((df["price"] < 0).sum()) if "price" in df.columns else 0)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 AI Readiness",
    "📊 Business Insight",
    "🏪 Merchant Insight",
    "🧹 Data Quality"
])

with tab1:
    st.subheader("Distribusi Label Kategori")

    freq = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(8,4))
    freq.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

    imbalance = freq.max()/freq.min()

    st.success(
        f"Rasio class imbalance: {imbalance:.2f}x"
    )

    low_class = freq[freq < 50]

    if len(low_class):
        st.warning(
            "Kategori dengan data rendah (<50 sampel): "
            + ", ".join(low_class.index.tolist())
        )

with tab2:

    st.subheader("Business Question 1")
    st.markdown(
        "**Kategori apa yang paling sering muncul sehingga menjadi prioritas utama model klasifikasi KantongKu?**"
    )

    top_cat = freq.reset_index()
    top_cat.columns = ["Kategori","Jumlah"]
    st.dataframe(top_cat, use_container_width=True)

    if "price" in df.columns:

        st.subheader("Business Question 2")
        st.markdown(
            "**Kategori mana yang memiliki rata-rata pengeluaran terbesar?**"
        )

        avg = (
            df.groupby("category")["price"]
            .mean()
            .sort_values(ascending=False)
        )

        fig2, ax2 = plt.subplots(figsize=(8,4))
        avg.sort_values().plot(kind="barh", ax=ax2)
        st.pyplot(fig2)

        st.dataframe(avg.reset_index(), use_container_width=True)

with tab3:

    if "merchant" in df.columns:

        st.subheader("Top 10 Merchant")

        topm = (
            df["merchant"]
            .value_counts()
            .head(10)
        )

        fig3, ax3 = plt.subplots(figsize=(8,4))
        topm.sort_values().plot(kind="barh", ax=ax3)
        st.pyplot(fig3)

        st.dataframe(topm.reset_index(), use_container_width=True)

        st.subheader("Merchant per Kategori")

        merchant_cat = (
            df.groupby("category")["merchant"]
            .nunique()
            .sort_values(ascending=False)
        )

        st.dataframe(
            merchant_cat.reset_index(),
            use_container_width=True
        )

with tab4:

    st.subheader("Missing Values")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Kolom","Missing"]

    st.dataframe(missing, use_container_width=True)

    st.subheader("Statistik Price")

    if "price" in df.columns:

        st.dataframe(
            df["price"].describe().to_frame(),
            use_container_width=True
        )

        fig4, ax4 = plt.subplots(figsize=(8,4))
        ax4.hist(df["price"].dropna(), bins=30)
        st.pyplot(fig4)

        negative = df[df["price"] < 0]

        if len(negative):
            st.warning(
                f"Ditemukan {len(negative)} transaksi bernilai negatif. "
                "Kemungkinan berasal dari refund, adjustment note, atau credit note."
            )

st.markdown("---")
st.caption(
    "KantongKu V3 • Fokus pada Data Quality, AI Readiness, dan Business Insight"
)
