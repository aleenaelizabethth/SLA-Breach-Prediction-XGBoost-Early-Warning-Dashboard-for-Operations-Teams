import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee SLA Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — dark industrial + amber palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:       #0d0e12;
    --surface:  #14161d;
    --card:     #1b1d27;
    --border:   #262933;
    --amber:    #f5a623;
    --amber2:   #ffcc70;
    --teal:     #2dd4bf;
    --rose:     #fb7185;
    --text:     #e8eaf0;
    --muted:    #7c8096;
    --radius:   12px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--amber), #e8920a);
    color: #0d0e12 !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(245,166,35,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245,166,35,0.4);
}

/* ── Selectbox / sliders / inputs ── */
div[data-baseweb="select"] > div,
.stSlider > div, .stNumberInput input, .stTextInput input {
    background-color: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] svg { fill: var(--amber) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--amber) !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: var(--radius); overflow: hidden; }
.stDataFrame thead tr th {
    background: var(--card) !important;
    color: var(--amber) !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
}
.stDataFrame tbody tr td { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.8rem 1.6rem;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
    background: transparent !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Custom info boxes ── */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.88rem;
    color: var(--muted);
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}
.predict-result-met {
    background: linear-gradient(135deg, #052e2e, #0a3d3d);
    border: 1px solid var(--teal);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
}
.predict-result-breached {
    background: linear-gradient(135deg, #2b0a12, #3d0f1c);
    border: 1px solid var(--rose);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
}
.predict-result-met h2, .predict-result-breached h2 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.3rem 0 0;
}
.predict-result-met h2 { color: var(--teal); }
.predict-result-breached h2 { color: var(--rose); }
.predict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#14161d",
    "axes.facecolor":    "#1b1d27",
    "axes.edgecolor":    "#262933",
    "axes.labelcolor":   "#e8eaf0",
    "xtick.color":       "#7c8096",
    "ytick.color":       "#7c8096",
    "text.color":        "#e8eaf0",
    "grid.color":        "#262933",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "axes.grid":         True,
    "axes.titlecolor":   "#e8eaf0",
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "font.family":       "sans-serif",
    "legend.facecolor":  "#1b1d27",
    "legend.edgecolor":  "#262933",
})
AMBER  = "#f5a623"
TEAL   = "#2dd4bf"
ROSE   = "#fb7185"
VIOLET = "#a78bfa"
PALETTE = [AMBER, TEAL, ROSE, VIOLET, "#34d399", "#60a5fa"]

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("employee_sla_status.csv")
    df.replace("", pd.NA, inplace=True)
    return df

# Paste study.best_params from your notebook here (leave {} to use XGBoost defaults)
BEST_PARAMS = {}

# Results copied from employee_project.ipynb (5-fold CV macro-F1, 3-fold CV for tuning)
CV_RESULTS = {"Logistic Regression": 0.823, "KNN": 0.863, "Random Forest": 0.981, "XGBoost": 0.981}
TUNING = {"GridSearchCV": 0.9764, "RandomizedSearchCV": 0.9775, "Optuna": 0.9799}
TASK_COLS = ["completion_ratio", "Tasks_Completed", "Tasks_Assigned"]

@st.cache_resource
def build_model(df: pd.DataFrame):
    d = df.copy()
    d["Met"] = (d["SLA_Status"] == "Met").astype(int)
    d["completion_ratio"] = d["Tasks_Completed"] / d["Tasks_Assigned"]
    d["was_missing"] = d["Avg_TAT_Hours"].isna().astype(int)
    d = pd.get_dummies(d.drop(columns=["Employee_ID", "SLA_Status"]),
                       columns=["Team", "Shift"], drop_first=True, dtype=int)

    X, y = d.drop(columns="Met"), d["Met"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    med = X_train.median()                       # medians from TRAIN only, no leakage
    X_train, X_test = X_train.fillna(med), X_test.fillna(med)

    model = XGBClassifier(random_state=42, n_jobs=-1, **BEST_PARAMS).fit(X_train, y_train)
    model_no_task = XGBClassifier(random_state=42, n_jobs=-1, **BEST_PARAMS).fit(
        X_train.drop(columns=TASK_COLS), y_train)   # honesty check
    return model, model_no_task, list(X.columns), X_test, y_test

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  `employee_sla_status.csv` not found. Place it in the same folder as `app.py`.")
    st.stop()

model, model_no_task, feature_cols, X_test, y_test = build_model(df)
y_pred = model.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average='macro')

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1rem 0 1.5rem">
            <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;
                        color:#f5a623;letter-spacing:-0.01em;line-height:1.2">
                Employee SLA<br>Intelligence
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:#7c8096;letter-spacing:0.08em;margin-top:0.3rem">
                XGBOOST SLA DASHBOARD
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    teams  = ["All"] + sorted(df["Team"].dropna().unique().tolist())
    shifts = ["All"] + sorted(df["Shift"].dropna().unique().tolist())
    sel_team  = st.selectbox("Filter by Team",  teams)
    sel_shift = st.selectbox("Filter by Shift", shifts)

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#7c8096">
        <b style="color:#e8eaf0">Dataset</b><br>
        {len(df):,} employees · {df.shape[1]} features<br><br>
        <b style="color:#e8eaf0">Model</b><br>
        XGBoost classifier<br>
        Test macro-F1: <span style="color:#f5a623;font-weight:700">
        {test_f1:.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# apply filters
fdf = df.copy()
if sel_team  != "All": fdf = fdf[fdf["Team"]  == sel_team]
if sel_shift != "All": fdf = fdf[fdf["Shift"] == sel_shift]

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="padding:2rem 0 1.5rem">
        <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                    letter-spacing:-0.03em;line-height:1.1;color:#e8eaf0">
            Employee SLA
            <span style="color:#f5a623">Status</span> Dashboard
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.92rem;
                    color:#7c8096;margin-top:0.5rem">
            Exploratory analysis · model comparison · risk flags · live prediction
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────────
total       = len(fdf)
met_count   = (fdf["SLA_Status"] == "Met").sum()
breach_count= (fdf["SLA_Status"] == "Breached").sum()
met_pct     = met_count / total * 100 if total else 0
teams_count = fdf["Team"].nunique()
avg_quality = fdf["Quality_Score"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Employees",  f"{total:,}")
k2.metric("SLA Met",          f"{met_count:,}",    f"{met_pct:.1f}%")
k3.metric("SLA Breached",     f"{breach_count:,}", f"{100-met_pct:.1f}%")
k4.metric("Teams",            f"{teams_count}")
k5.metric("Avg Quality Score",f"{avg_quality:.1f}" if not np.isnan(avg_quality) else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋  Data Overview", "📈  EDA & Insights", "🤖  Model Performance", "🚩  Risk Flags", "🔮  Predict SLA"]
)

# ══════════════════════════════════════════════
#  TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Raw Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">FILTERED VIEW · SCROLL TO EXPLORE</div>', unsafe_allow_html=True)
    st.dataframe(fdf, use_container_width=True, height=320)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">NUMERIC COLUMNS ONLY</div>', unsafe_allow_html=True)
        st.dataframe(fdf.describe().round(2), use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">COUNT & PERCENTAGE PER COLUMN</div>', unsafe_allow_html=True)
        miss = fdf.isna().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["Pct (%)"] = (miss["Missing"] / len(fdf) * 100).round(2)
        miss = miss[miss["Missing"] > 0]
        if miss.empty:
            st.markdown(
                '<div class="info-card">✅ No missing values in the current filter selection.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(miss, use_container_width=True)

    # Data types strip
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Column Data Types</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">SCHEMA OVERVIEW</div>', unsafe_allow_html=True)
    dtypes_df = pd.DataFrame(
        {"Column": fdf.dtypes.index, "Dtype": fdf.dtypes.values.astype(str)}
    )
    st.dataframe(dtypes_df, use_container_width=True, height=200)

# ══════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    row1_l, row1_r = st.columns(2)

    # ── Employees per Team ──
    with row1_l:
        st.markdown('<div class="section-header">Employees per Team</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">HEADCOUNT DISTRIBUTION</div>', unsafe_allow_html=True)
        team_counts = fdf["Team"].value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        bars = ax.bar(team_counts.index, team_counts.values,
                      color=PALETTE[:len(team_counts)], width=0.55,
                      edgecolor="#0d0e12", linewidth=0.8)
        for bar, val in zip(bars, team_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    str(val), ha="center", va="bottom", fontsize=9, color="#e8eaf0")
        ax.set_xlabel("Team", labelpad=8)
        ax.set_ylabel("Headcount")
        ax.set_title("Employees per Team")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Shift Distribution ──
    with row1_r:
        st.markdown('<div class="section-header">Shift Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">DAY VS NIGHT WORKFORCE</div>', unsafe_allow_html=True)
        shift_counts = fdf["Shift"].value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        wedges, texts, autotexts = ax.pie(
            shift_counts.values,
            labels=shift_counts.index,
            autopct="%1.1f%%",
            colors=[AMBER, TEAL],
            startangle=90,
            wedgeprops=dict(edgecolor="#0d0e12", linewidth=1.5, width=0.65),
            pctdistance=0.75,
        )
        for t in autotexts: t.set_color("#0d0e12"); t.set_fontweight("bold")
        for t in texts: t.set_color("#e8eaf0")
        ax.set_title("Shift Distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    row2_l, row2_r = st.columns(2)

    # ── SLA Status ──
    with row2_l:
        st.markdown('<div class="section-header">SLA Status Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">MET VS BREACHED</div>', unsafe_allow_html=True)
        sla_counts = fdf["SLA_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        bars = ax.bar(sla_counts.index, sla_counts.values,
                      color=[ROSE, TEAL][:len(sla_counts)], width=0.45,
                      edgecolor="#0d0e12", linewidth=0.8)
        for bar, val in zip(bars, sla_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="#e8eaf0")
        ax.set_xlabel("SLA Status"); ax.set_ylabel("Count")
        ax.set_title("SLA Status Distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── SLA by Shift ──
    with row2_r:
        st.markdown('<div class="section-header">SLA Status by Shift</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">SHIFT-LEVEL BREAKDOWN</div>', unsafe_allow_html=True)
        shift_sla = pd.crosstab(fdf["Shift"], fdf["SLA_Status"])
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        x = np.arange(len(shift_sla))
        w = 0.35
        for i, (col, color) in enumerate(zip(shift_sla.columns, [ROSE, TEAL])):
            ax.bar(x + i * w, shift_sla[col], width=w, label=col,
                   color=color, edgecolor="#0d0e12", linewidth=0.8)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(shift_sla.index)
        ax.set_ylabel("Count")
        ax.set_title("SLA Status by Shift")
        ax.legend(frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation Heatmap ──
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">NUMERIC FEATURES · LOWER TRIANGLE</div>', unsafe_allow_html=True)
    cor = fdf.corr(numeric_only=True)
    mask = np.triu(np.ones_like(cor, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        cor, mask=mask, annot=True, cmap="YlOrBr", center=0,
        fmt=".2f", linewidths=0.4, linecolor="#0d0e12",
        cbar_kws={"shrink": 0.75}, ax=ax,
    )
    ax.set_title("Correlation Heatmap", pad=14)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Boxplot Outlier Detection ──
    st.markdown('<div class="section-header">Outlier Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">BOX PLOTS · NUMERIC FEATURES</div>', unsafe_allow_html=True)
    num_cols = fdf.select_dtypes(include=["int64", "float64"]).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(
        [fdf[c].dropna().values for c in num_cols],
        patch_artist=True,
        medianprops=dict(color=AMBER, linewidth=2),
        whiskerprops=dict(color="#7c8096"),
        capprops=dict(color="#7c8096"),
        flierprops=dict(marker="o", markerfacecolor=ROSE, markersize=3, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], PALETTE * 5):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(num_cols, rotation=30, ha="right")
    ax.set_title("Outlier Detection — Numeric Features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quality Score Distribution ──
    st.markdown('<div class="section-header">Quality Score Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">HISTOGRAM BY SLA STATUS</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    for status, color in [("Breached", ROSE), ("Met", TEAL)]:
        subset = fdf[fdf["SLA_Status"] == status]["Quality_Score"].dropna()
        ax.hist(subset, bins=20, alpha=0.65, color=color, label=status, edgecolor="#0d0e12")
    ax.set_xlabel("Quality Score"); ax.set_ylabel("Frequency")
    ax.set_title("Quality Score Distribution by SLA Status")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">XGBoost Classifier — Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">80/20 STRATIFIED SPLIT · IMPUTED WITH TRAIN MEDIANS · SCORED ON UNSEEN TEST DATA</div>',
                unsafe_allow_html=True)

    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["Breached", "Met"], output_dict=True)
    f1_no_task = f1_score(y_test, model_no_task.predict(X_test.drop(columns=TASK_COLS)), average="macro")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Macro-F1", f"{test_f1:.3f}")
    m2.metric("Accuracy", f"{acc*100:.1f}%")
    m3.metric("Recall (Breached)", f"{rep['Breached']['recall']*100:.1f}%")
    m4.metric("Macro-F1 without task columns", f"{f1_no_task:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">5-FOLD CV MACRO-F1 · FROM NOTEBOOK</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        names, vals = list(CV_RESULTS.keys()), list(CV_RESULTS.values())
        bars = ax.bar(names, vals, color=[VIOLET, VIOLET, AMBER, AMBER], edgecolor="#0d0e12")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Macro-F1")
        plt.xticks(rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Tuning Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">XGBOOST · 3-FOLD CV MACRO-F1 · FROM NOTEBOOK</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Method": list(TUNING.keys()), "Macro-F1": list(TUNING.values())}),
                     use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="info-card">All three methods land within <b>0.004</b> of each other, and untuned '
            'XGBoost already scores about 0.98. <b>Tuning added almost nothing</b> on this dataset.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="info-card">Tree models beat Logistic Regression and KNN by 0.12 to 0.16 macro-F1. '
            'Feature engineering (<b>completion ratio</b>) and a leak-free split mattered more than tuning.</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(cm, cmap="YlOrBr", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Breached", "Met"]); ax.set_yticklabels(["Breached", "Met"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=20, fontweight="bold",
                        color="#0d0e12" if cm[i, j] > cm.max() / 2 else "#e8eaf0")
        plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_fi:
        st.markdown('<div class="section-header">What Drives the Prediction</div>', unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.barh(imp.index, imp.values, color=AMBER)
        ax.set_xlabel("Feature importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(
        '<div class="info-card"><b>Limitation:</b> a score this close to perfect suggests the SLA label may follow '
        'rules built from these columns, so the model is recovering a pattern. It is not evidence of real-world '
        'accuracy. The dataset also looks synthetic.</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — RISK FLAGS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Risk Flags</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">WHO A MANAGER SHOULD REVIEW FIRST · TEST SET</div>', unsafe_allow_html=True)

    thr = st.slider("Flag employees whose breach risk is at or above", 0.30, 0.95, 0.70, 0.05)
    risk = 1 - model.predict_proba(X_test)[:, 1]
    flagged = risk >= thr
    breached = (y_test.values == 0)

    f1c, f2c, f3c = st.columns(3)
    f1c.metric("Employees flagged", f"{int(flagged.sum()):,}")
    f2c.metric("Flagged who really breached", f"{breached[flagged].mean()*100:.1f}%" if flagged.any() else "—")
    f3c.metric("Breaches caught", f"{flagged[breached].mean()*100:.1f}%")
    st.markdown(
        '<div class="info-card">A lower cutoff catches more breaches but flags more people. '
        'Pick the cutoff to match how many reviews a manager can handle.</div>', unsafe_allow_html=True)

    ranked = pd.DataFrame({"Risk": risk}, index=X_test.index).sort_values("Risk", ascending=False).head(15)
    table = df.loc[ranked.index, ["Team", "Shift", "Tasks_Assigned", "Tasks_Completed",
                                  "Avg_TAT_Hours", "Quality_Score", "SLA_Status"]].copy()
    table.insert(0, "Risk %", (ranked["Risk"] * 100).round(1))
    st.markdown('<div class="section-header">Top 15 Highest-Risk Records</div>', unsafe_allow_html=True)
    st.dataframe(table, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 5 — PREDICT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Predict Employee SLA Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ENTER EMPLOYEE METRICS · GET INSTANT PREDICTION</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tasks_assigned  = st.number_input("Tasks Assigned",  min_value=1, max_value=500, value=120, step=1)
            tasks_completed = st.number_input("Tasks Completed", min_value=0, max_value=500, value=110, step=1)
            avg_tat         = st.number_input("Avg TAT (Hours)", min_value=0.0, max_value=20.0, value=4.5, step=0.1, format="%.2f")
        with c2:
            error_rate    = st.number_input("Error Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, format="%.2f")
            quality_score = st.number_input("Quality Score",  min_value=0.0, max_value=100.0, value=80.0, step=0.1, format="%.2f")
            escalations   = st.number_input("Escalations",    min_value=0, max_value=50, value=2, step=1)
        with c3:
            rework_count     = st.number_input("Rework Count",        min_value=0, max_value=50, value=3, step=1)
            experience_years = st.number_input("Experience (Years)",  min_value=0.0, max_value=30.0, value=3.0, step=0.1, format="%.1f")
            team_input  = st.selectbox("Team",  ["Ads", "Moderation", "QA", "Support"])
            shift_input = st.selectbox("Shift", ["Day", "Night"])
        submitted = st.form_submit_button("⚡  Predict SLA Status", use_container_width=True)

    if submitted:
        row = {c: 0 for c in feature_cols}
        row.update({
            "Tasks_Assigned": tasks_assigned, "Tasks_Completed": tasks_completed,
            "Avg_TAT_Hours": avg_tat, "Error_Rate": error_rate, "Quality_Score": quality_score,
            "Escalations": escalations, "Rework_Count": rework_count, "Experience_Years": experience_years,
            "completion_ratio": tasks_completed / tasks_assigned, "was_missing": 0,
        })
        if team_input != "Ads":
            row["Team_" + team_input] = 1
        if shift_input == "Night":
            row["Shift_Night"] = 1

        input_df = pd.DataFrame([row])[feature_cols]
        proba = model.predict_proba(input_df)[0]          # [Breached, Met]
        pred = int(proba.argmax())
        breach_risk = proba[0]

        st.markdown("<br>", unsafe_allow_html=True)
        res_col, prob_col = st.columns([1.2, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                    <div class="predict-result-met">
                        <div class="predict-label">PREDICTION RESULT</div>
                        <h2>✅ SLA MET</h2>
                        <div style="color:#7c8096;font-size:0.82rem;margin-top:0.5rem;font-family:'DM Mono',monospace">
                            Breach risk {breach_risk*100:.0f}% · completion ratio {tasks_completed/tasks_assigned:.2f}
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="predict-result-breached">
                        <div class="predict-label">PREDICTION RESULT</div>
                        <h2>⚠️ SLA BREACHED</h2>
                        <div style="color:#7c8096;font-size:0.82rem;margin-top:0.5rem;font-family:'DM Mono',monospace">
                            Breach risk {breach_risk*100:.0f}% · completion ratio {tasks_completed/tasks_assigned:.2f}
                        </div>
                    </div>""", unsafe_allow_html=True)

        with prob_col:
            st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 3))
            bars = ax.barh(["Breached", "Met"], proba * 100, color=[ROSE, TEAL], height=0.45, edgecolor="#0d0e12")
            for bar, val in zip(bars, proba * 100):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                        va="center", fontsize=10, fontweight="bold", color="#e8eaf0")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;font-family:'DM Mono',monospace;font-size:0.7rem;
                color:#3a3d4d;padding:1rem 0">
        Employee SLA Intelligence Dashboard · XGBoost Classifier · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
