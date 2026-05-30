import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KantongKu – EDA Dashboard",
    page_icon="👛",
    layout="wide",
)

KATEGORI_ORDER = [
    "Belanja", "Makanan & Minuman", "Lain-lain",
    "Edukasi", "Jajan", "Kesehatan", "Transportasi", "Hiburan"
]

PALETTE = {
    "Belanja":           "#4C72B0",
    "Makanan & Minuman": "#DD8452",
    "Lain-lain":         "#55A868",
    "Edukasi":           "#C44E52",
    "Jajan":             "#8172B2",
    "Kesehatan":         "#937860",
    "Transportasi":      "#DA8BC3",
    "Hiburan":           "#8C8C8C",
    "Makan dan Minuman": "#DD8452",   # alias dari kaggle
}

MYR_TO_IDR = 4489  # kurs Mei 2026

# ─────────────────────────────────────────────
# HARDCODED DATA (fallback dari hasil EDA)
# ─────────────────────────────────────────────
HARDCODED = pd.DataFrame({
    "category": KATEGORI_ORDER,
    "jumlah":   [301, 255, 145, 111, 111, 45, 30, 18],
    "avg_price":[31860579, 9886171, 36864863, 8472471,
                 12398933, 34870858, 13113910, 44923139],
})

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt_rupiah(val):
    return f"Rp {val:,.0f}".replace(",", ".")

def normalize_category(s):
    mapping = {"Makan dan Minuman": "Makanan & Minuman"}
    return mapping.get(s, s)

def load_and_process(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # normalise kolom
    df.columns = [c.strip().lower() for c in df.columns]
    if "category" not in df.columns:
        raise ValueError("Kolom 'category' tidak ditemukan di CSV.")
    df["category"] = df["category"].apply(normalize_category)
    # konversi harga jika ada kolom price
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "filename" in df.columns:
            mask_kaggle = ~df["filename"].str.startswith("nota_")
            df.loc[mask_kaggle, "price"] = df.loc[mask_kaggle, "price"] * MYR_TO_IDR
    return df

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/purse-emoji.png", width=64)
    st.title("👛 KantongKu")
    st.caption("Coding Camp 2026 – CC26-PSU398")
    st.divider()

    st.subheader("📂 Sumber Data")
    mode = st.radio(
        "Pilih mode data:",
        ["Gunakan Data Bawaan (Hardcoded)", "Upload CSV (final_dataset_combined.csv)"],
    )

    uploaded = None
    if mode == "Upload CSV (final_dataset_combined.csv)":
        uploaded = st.file_uploader("Upload file CSV", type=["csv"])
        if not uploaded:
            st.info("Belum ada file. Menampilkan data bawaan sebagai preview.")

    st.divider()
    st.caption("Dataset: Struk Lokal Indonesia + Malaysia\nPeriode: 2025–2026\nTotal: 1.016 transaksi")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_raw = None
use_raw = False

if uploaded:
    try:
        df_raw = load_and_process(uploaded)
        use_raw = True
        st.sidebar.success(f"✅ {len(df_raw)} baris berhasil dimuat.")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca CSV: {e}")

if use_raw and df_raw is not None:
    freq = df_raw["category"].value_counts().reset_index()
    freq.columns = ["category", "jumlah"]
    if "price" in df_raw.columns:
        avg_price = df_raw.groupby("category")["price"].mean().reset_index()
        avg_price.columns = ["category", "avg_price"]
        stats = freq.merge(avg_price, on="category", how="left")
    else:
        stats = freq.copy()
        stats["avg_price"] = None
else:
    stats = HARDCODED.copy()
    df_raw = None

# urutkan sesuai KATEGORI_ORDER kalau ada
ordered = [k for k in KATEGORI_ORDER if k in stats["category"].values]
others  = [k for k in stats["category"].values if k not in KATEGORI_ORDER]
stats["category"] = pd.Categorical(stats["category"], categories=ordered + others, ordered=True)
stats = stats.sort_values("category").reset_index(drop=True)

total_transaksi = stats["jumlah"].sum()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("👛 KantongKu – EDA Dashboard")
st.caption("Exploratory Data Analysis · Dataset Struk Gabungan (Lokal Indonesia + Malaysia) · 2025–2026")
st.divider()

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transaksi", f"{total_transaksi:,}".replace(",","."))
k2.metric("Jumlah Kategori", str(len(stats)))
k3.metric("Kategori Terbanyak", stats.loc[stats["jumlah"].idxmax(), "category"])
if stats["avg_price"].notna().any():
    k4.metric("Avg Tertinggi", stats.loc[stats["avg_price"].idxmax(), "category"])
else:
    k4.metric("Avg Tertinggi", "—")

st.divider()

# ─────────────────────────────────────────────
# PERTANYAAN BISNIS 1
# ─────────────────────────────────────────────
st.subheader("📊 Pertanyaan Bisnis 1")
st.markdown(
    "> **Kategori pengeluaran apa yang paling sering muncul pada dataset struk gabungan "
    "(Lokal Indonesia + Malaysia) periode 2025–2026, sehingga dapat menentukan prioritas "
    "kelas label model klasifikasi KantongKu?**"
)

col_chart1, col_insight1 = st.columns([2, 1])

with col_chart1:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors1 = [PALETTE.get(c, "#999999") for c in stats["category"]]
    bars = ax1.barh(stats["category"], stats["jumlah"], color=colors1, edgecolor="white")
    ax1.bar_label(bars, padding=4, fontsize=9)
    ax1.set_xlabel("Jumlah Transaksi", fontsize=10)
    ax1.set_title("Distribusi Frekuensi per Kategori", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col_insight1:
    st.markdown("**📌 Temuan:**")
    top3 = stats.nlargest(3, "jumlah")
    for _, row in top3.iterrows():
        pct = row["jumlah"] / total_transaksi * 100
        st.metric(row["category"], f"{int(row['jumlah'])} transaksi", f"{pct:.1f}% dari total")

    st.divider()
    st.markdown("**⚠️ Class Imbalance:**")
    max_j = stats["jumlah"].max()
    min_j = stats["jumlah"].min()
    rasio = max_j / min_j
    st.warning(
        f"Selisih antara kategori terbanyak **({int(max_j)})** "
        f"dan tersedikit **({int(min_j)})** mencapai **{rasio:.0f}x lipat**. "
        f"Perlu penanganan saat training (SMOTE / class weighting)."
    )

# Donut chart
st.markdown("")
col_donut, col_tabel = st.columns([1, 1])

with col_donut:
    fig_d, ax_d = plt.subplots(figsize=(5, 5))
    colors_d = [PALETTE.get(c, "#999999") for c in stats["category"]]
    wedges, texts, autotexts = ax_d.pie(
        stats["jumlah"],
        labels=stats["category"],
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        colors=colors_d,
        startangle=140,
        wedgeprops=dict(width=0.5),
        textprops={"fontsize": 8},
    )
    ax_d.set_title("Proporsi Kategori (%)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close()

with col_tabel:
    st.markdown("**Tabel Distribusi Kategori**")
    tbl1 = stats[["category", "jumlah"]].copy()
    tbl1["persentase"] = (tbl1["jumlah"] / total_transaksi * 100).round(1).astype(str) + "%"
    tbl1.columns = ["Kategori", "Jumlah Transaksi", "Persentase"]
    st.dataframe(tbl1, use_container_width=True, hide_index=True)

st.success(
    "**Kesimpulan PB1:** Kategori **Belanja** mendominasi dengan "
    f"{stats[stats['category']=='Belanja']['jumlah'].values[0] if 'Belanja' in stats['category'].values else '—'} "
    "transaksi, disusul **Makanan & Minuman** dan **Lain-lain**. "
    "Ketiga kategori ini mencakup >69% total data dan menjadi prioritas label utama model klasifikasi."
)

st.divider()

# ─────────────────────────────────────────────
# PERTANYAAN BISNIS 2
# ─────────────────────────────────────────────
st.subheader("💰 Pertanyaan Bisnis 2")
st.markdown(
    "> **Kategori pengeluaran mana yang memiliki rata-rata nominal tertinggi dalam Rupiah "
    "pada dataset struk gabungan periode 2025–2026, sehingga dapat memberikan gambaran "
    "pola pengeluaran sebagai referensi rekomendasi alokasi kantong anggaran awal?**"
)

if stats["avg_price"].notna().any():
    stats_price = stats.dropna(subset=["avg_price"]).sort_values("avg_price", ascending=False)

    col_bar2, col_insight2 = st.columns([2, 1])

    with col_bar2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors2 = [PALETTE.get(c, "#999999") for c in stats_price["category"]]
        bars2 = ax2.barh(stats_price["category"], stats_price["avg_price"], color=colors2, edgecolor="white")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}jt"))
        ax2.bar_label(
            bars2,
            labels=[f"Rp {v/1e6:.1f}jt" for v in stats_price["avg_price"]],
            padding=4, fontsize=8
        )
        ax2.set_xlabel("Rata-rata Pengeluaran (IDR)", fontsize=10)
        ax2.set_title("Rata-rata Nominal per Kategori", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_insight2:
        st.markdown("**📌 Temuan:**")
        top3p = stats_price.head(3)
        for _, row in top3p.iterrows():
            st.metric(row["category"], fmt_rupiah(row["avg_price"]))

        st.divider()
        st.markdown("**💡 Pola Menarik:**")
        st.info(
            "**Hiburan** → jarang tapi mahal per transaksi.\n\n"
            "**Makanan & Minuman** → sering tapi nominal kecil → pengeluaran rutin harian."
        )

    # Scatter: frekuensi vs avg price
    st.markdown("**Scatter: Frekuensi vs Rata-rata Nominal**")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    for _, row in stats_price.iterrows():
        c = PALETTE.get(row["category"], "#999999")
        ax3.scatter(row["jumlah"], row["avg_price"], color=c, s=120, zorder=3)
        ax3.annotate(
            row["category"],
            (row["jumlah"], row["avg_price"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8,
        )
    ax3.set_xlabel("Frekuensi Transaksi", fontsize=10)
    ax3.set_ylabel("Rata-rata Nominal (IDR)", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}jt"))
    ax3.set_title("Frekuensi vs Rata-rata Nominal per Kategori", fontsize=12, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Tabel lengkap
    st.markdown("**Tabel Rata-rata Nominal per Kategori**")
    tbl2 = stats_price[["category", "avg_price", "jumlah"]].copy()
    tbl2["avg_price_fmt"] = tbl2["avg_price"].apply(fmt_rupiah)
    tbl2 = tbl2[["category", "avg_price_fmt", "jumlah"]]
    tbl2.columns = ["Kategori", "Rata-rata Pengeluaran (IDR)", "Jumlah Transaksi"]
    st.dataframe(tbl2, use_container_width=True, hide_index=True)

    st.success(
        "**Kesimpulan PB2:** Kategori **Hiburan** memiliki rata-rata nominal tertinggi "
        f"({fmt_rupiah(stats_price[stats_price['category']=='Hiburan']['avg_price'].values[0]) if 'Hiburan' in stats_price['category'].values else '—'}) "
        "meski frekuensinya paling rendah. Ini menjadi dasar rekomendasi alokasi kantong anggaran awal KantongKu."
    )
else:
    st.info("Upload `final_dataset_combined.csv` (yang memiliki kolom `price`) untuk melihat analisis rata-rata nominal.")

st.divider()

# ─────────────────────────────────────────────
# BONUS: TREND BULANAN (jika ada kolom date)
# ─────────────────────────────────────────────
if df_raw is not None and "date" in df_raw.columns and "price" in df_raw.columns:
    st.subheader("📅 Bonus: Trend Pengeluaran Bulanan")
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw["month"] = df_raw["date"].dt.to_period("M")
    monthly = df_raw.groupby("month")["price"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    monthly = monthly.sort_values("month")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(monthly["month_str"], monthly["price"], marker="o", color="#4C72B0", linewidth=2)
    ax4.fill_between(range(len(monthly)), monthly["price"], alpha=0.1, color="#4C72B0")
    ax4.set_xticks(range(len(monthly)))
    ax4.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e9:.1f}M"))
    ax4.set_title("Total Pengeluaran per Bulan", fontsize=12, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.divider()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.caption(
    "KantongKu · Coding Camp 2026 powered by DBS Foundation · Team CC26-PSU398  \n"
    "Data Science: Muhammad Davin Al Hisyam & Muhammad Alwi Marom"
)
