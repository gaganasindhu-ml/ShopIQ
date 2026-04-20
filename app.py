import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ShopIQ — Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0F1117; }

.hero-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-title span { color: #A3E635; }
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(163,230,53,0.15);
    border: 1px solid rgba(163,230,53,0.3);
    color: #A3E635;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.metric-card {
    background: #1e2130;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #A3E635;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.seg-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}
.seg-name { font-size: 1rem; font-weight: 600; color: #fff; }
.seg-stats { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
.seg-desc { font-size: 0.82rem; color: #cbd5e1; margin-top: 6px; line-height: 1.5; }
.seg-strategy {
    font-size: 0.78rem;
    color: #A3E635;
    margin-top: 8px;
    padding: 6px 10px;
    background: rgba(163,230,53,0.08);
    border-radius: 6px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d3561;
}

.stTabs [data-baseweb="tab-list"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #2d3561 !important;
    color: #A3E635 !important;
}

.sidebar-info {
    background: #1e2130;
    border: 1px solid #2d3561;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.sidebar-value {
    font-size: 1rem;
    font-weight: 600;
    color: #A3E635;
}
</style>
""", unsafe_allow_html=True)

# ── Colours for 5 segments ────────────────────────────────────
SEG_COLORS  = ['#1D9E75','#534AB7','#D85A30','#EF9F27','#888780']
SEG_NAMES   = ['Average Joes','Premium Targets','Impulsive Buyers','Rich but Reluctant','Careful Savers']
SEG_ICONS   = ['👤','⭐','🛒','💼','🪙']
SEG_DESC    = [
    'Middle income, middle spending. The typical everyday customer.',
    'High income AND high spending. The dream customer — highest value.',
    'Low income but HIGH spending. Loyal and impulsive buyers.',
    'High income, barely spending. Biggest untapped opportunity.',
    'Low income, low spending. Window shoppers and necessity buyers.'
]
SEG_STRATEGY = [
    'Seasonal offers, bundle deals, family packages',
    'VIP program, premium brands, exclusive previews',
    'Flash sales, EMI options, loyalty points',
    'Premium events, surveys, exclusive experiences',
    'Discount coupons, clearance sales'
]
SEG_BG = [
    'rgba(29,158,117,0.12)','rgba(83,74,183,0.12)',
    'rgba(216,90,48,0.12)','rgba(239,159,39,0.12)','rgba(136,135,128,0.12)'
]
SEG_BORDER = ['#1D9E75','#534AB7','#D85A30','#EF9F27','#888780']

# ── Load + process data ───────────────────────────────────────
@st.cache_data
def load_and_process(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    # normalise column names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if 'income' in cl:   rename[c] = 'Annual Income (k$)'
        if 'spending' in cl: rename[c] = 'Spending Score'
        if 'age' in cl:      rename[c] = 'Age'
        if 'gender' in cl:   rename[c] = 'Gender'
        if 'id' in cl:       rename[c] = 'CustomerID'
    df = df.rename(columns=rename)

    X = df[['Annual Income (k$)','Spending Score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X_scaled)
    df['Cluster'] = km.labels_

    # Remap clusters to consistent segment names based on centroid position
    centers = scaler.inverse_transform(km.cluster_centers_)
    order = {}
    for i, (inc, spd) in enumerate(centers):
        if inc < 50 and spd < 50:   order[i] = 4   # Careful Savers
        elif inc < 50 and spd >= 50: order[i] = 2  # Impulsive Buyers
        elif inc >= 50 and spd >= 50 and inc < 75: order[i] = 0  # Average Joes
        elif inc >= 75 and spd >= 50: order[i] = 1  # Premium Targets
        else: order[i] = 3                           # Rich but Reluctant
    df['Segment'] = df['Cluster'].map(order)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    df['Anomaly'] = iso.predict(X_scaled)

    return df, X, X_scaled, scaler, km, centers

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ **ShopIQ**")
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:1.5rem'>AI Customer Segmentation</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your CSV", type=['csv'],
        help="Must have: Annual Income, Spending Score columns")

    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Default Dataset</div>", unsafe_allow_html=True)
    use_default = st.checkbox("Use Mall Customers dataset", value=True)

    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Model Settings</div>", unsafe_allow_html=True)
    n_clusters = st.slider("Number of segments (K)", 2, 8, 5)
    contamination = st.slider("Anomaly sensitivity", 0.01, 0.20, 0.05, 0.01,
        help="Fraction of customers expected to be unusual")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#64748b;line-height:1.8'>
    <b style='color:#94a3b8'>Algorithms used</b><br>
    K-Means Clustering<br>
    Isolation Forest<br>
    PCA (visualisation)<br>
    StandardScaler
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
try:
    if uploaded:
        df, X, X_scaled, scaler, km, centers = load_and_process(uploaded)
    else:
        df, X, X_scaled, scaler, km, centers = load_and_process('Mall_Customers.csv')
except Exception as e:
    st.error(f"Error loading data: {e}. Please upload a CSV with Annual Income and Spending Score columns.")
    st.stop()

anomaly_count = (df['Anomaly'] == -1).sum()
total = len(df)

# ── HERO ──────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-card'>
  <div class='hero-badge'>● AI-POWERED · RETAIL ANALYTICS</div>
  <div class='hero-title'>Shop<span>IQ</span></div>
  <div style='font-size:1.4rem;color:#94a3b8;font-weight:400;margin-top:4px'>Customer Segmentation Engine</div>
  <div class='hero-sub'>K-Means model trained on {total} customer records · {n_clusters} segments · Powered by sklearn</div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{total}</div>
        <div class='metric-label'>Total Customers</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>5</div>
        <div class='metric-label'>Segments Found</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{anomaly_count}</div>
        <div class='metric-label'>Anomalies Flagged</div>
    </div>""", unsafe_allow_html=True)
with m4:
    avg_spend = df['Spending Score'].mean()
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{avg_spend:.0f}</div>
        <div class='metric-label'>Avg Spending Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Segments", "📊 EDA", "🔴 Anomalies", "📈 Data"
])

plt.style.use('dark_background')
PLOT_BG   = '#1e2130'
PLOT_FG   = '#e2e8f0'
GRID_COL  = '#2d3561'

# ══════════════════════════════════════════════
# TAB 1 — SEGMENTS
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-title'>Customer Segments Map</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)

        for i in range(5):
            mask = df['Segment'] == i
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=SEG_COLORS[i], label=f'{SEG_ICONS[i]} {SEG_NAMES[i]}',
                       s=60, alpha=0.85, edgecolors='white', linewidth=0.3)

        real_centers = scaler.inverse_transform(km.cluster_centers_)
        ax.scatter(real_centers[:, 0], real_centers[:, 1],
                   c='white', marker='*', s=280, zorder=10, label='Centroids')

        ax.set_xlabel('Annual Income (k$)', color=PLOT_FG, fontsize=10)
        ax.set_ylabel('Spending Score', color=PLOT_FG, fontsize=10)
        ax.set_title('Mall Customer Segments — K-Means K=5', color=PLOT_FG, fontsize=11, fontweight='bold')
        ax.tick_params(colors=PLOT_FG)
        ax.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values(): spine.set_color(GRID_COL)
        leg = ax.legend(fontsize=8, facecolor=PLOT_BG, edgecolor=GRID_COL,
                        labelcolor=PLOT_FG, loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("<div class='section-title'>Segment Profiles</div>", unsafe_allow_html=True)

        summary = df.groupby('Segment').agg(
            Income=('Annual Income (k$)', 'mean'),
            Spending=('Spending Score', 'mean'),
            Count=('Segment', 'count')
        ).round(1)

        for i in range(5):
            if i in summary.index:
                row = summary.loc[i]
                st.markdown(f"""
                <div class='seg-card' style='background:{SEG_BG[i]};border-color:{SEG_BORDER[i]}'>
                  <div class='seg-name'>{SEG_ICONS[i]} {SEG_NAMES[i]}</div>
                  <div class='seg-stats'>Income: <b>{row.Income:.0f}k</b> &nbsp;·&nbsp; Spending: <b>{row.Spending:.0f}</b> &nbsp;·&nbsp; {row.Count:.0f} customers</div>
                  <div class='seg-desc'>{SEG_DESC[i]}</div>
                  <div class='seg-strategy'>💡 {SEG_STRATEGY[i]}</div>
                </div>
                """, unsafe_allow_html=True)

    # Elbow chart
    st.markdown("<div class='section-title'>Elbow Method — How K=5 was chosen</div>", unsafe_allow_html=True)
    inertias = []
    for k in range(1, 11):
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor(PLOT_BG)
    ax2.set_facecolor(PLOT_BG)
    ax2.plot(range(1, 11), inertias, 'o-', color='#534AB7', linewidth=2.5, markersize=7)
    ax2.axvline(x=5, color='#A3E635', linestyle='--', linewidth=1.5, label='Optimal K=5')
    ax2.set_xlabel('Number of clusters K', color=PLOT_FG)
    ax2.set_ylabel('Inertia', color=PLOT_FG)
    ax2.set_title('Elbow Method', color=PLOT_FG, fontweight='bold')
    ax2.tick_params(colors=PLOT_FG)
    ax2.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)
    for spine in ax2.spines.values(): spine.set_color(GRID_COL)
    ax2.legend(fontsize=9, facecolor=PLOT_BG, edgecolor=GRID_COL, labelcolor=PLOT_FG)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Gender distribution
        if 'Gender' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)
            counts = df['Gender'].value_counts()
            bars = ax.bar(counts.index, counts.values,
                          color=['#534AB7','#D85A30'], edgecolor='white', linewidth=0.5)
            ax.set_title('Gender Distribution', color=PLOT_FG, fontweight='bold')
            ax.tick_params(colors=PLOT_FG)
            ax.grid(True, color=GRID_COL, alpha=0.4, axis='y')
            for spine in ax.spines.values(): spine.set_color(GRID_COL)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        str(int(bar.get_height())), ha='center', color=PLOT_FG, fontsize=10)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        # Age distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)
        ax.hist(df['Age'], bins=15, color='#1D9E75', edgecolor='white', alpha=0.85)
        ax.set_title('Age Distribution', color=PLOT_FG, fontweight='bold')
        ax.set_xlabel('Age', color=PLOT_FG); ax.set_ylabel('Count', color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        ax.grid(True, color=GRID_COL, alpha=0.4, axis='y')
        for spine in ax.spines.values(): spine.set_color(GRID_COL)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        # Income distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)
        ax.hist(df['Annual Income (k$)'], bins=15, color='#EF9F27', edgecolor='white', alpha=0.85)
        ax.set_title('Income Distribution', color=PLOT_FG, fontweight='bold')
        ax.set_xlabel('Annual Income (k$)', color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        ax.grid(True, color=GRID_COL, alpha=0.4, axis='y')
        for spine in ax.spines.values(): spine.set_color(GRID_COL)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        # Spending distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)
        ax.hist(df['Spending Score'], bins=15, color='#D85A30', edgecolor='white', alpha=0.85)
        ax.set_title('Spending Score Distribution', color=PLOT_FG, fontweight='bold')
        ax.set_xlabel('Spending Score', color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        ax.grid(True, color=GRID_COL, alpha=0.4, axis='y')
        for spine in ax.spines.values(): spine.set_color(GRID_COL)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
    num_cols = ['Age','Annual Income (k$)','Spending Score']
    corr = df[num_cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, ax=ax, annot_kws={'size':11,'color':'white'})
    ax.tick_params(colors=PLOT_FG)
    ax.set_title('Feature Correlations', color=PLOT_FG, fontweight='bold')
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.info("💡 Key finding: Income vs Spending correlation = **0.01** — wealthy customers do NOT automatically spend more. This is why 5 distinct segments exist.")

# ══════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Anomaly Detection — Isolation Forest</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1], gap="large")

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PLOT_BG); ax.set_facecolor(PLOT_BG)

        for i in range(5):
            mask = (df['Segment'] == i) & (df['Anomaly'] == 1)
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=SEG_COLORS[i], s=55, alpha=0.65,
                       edgecolors='white', linewidth=0.3)

        anomaly_mask = df['Anomaly'] == -1
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
                   c='#E24B4A', s=160, marker='X', zorder=10,
                   label=f'Anomaly ({anomaly_mask.sum()} customers)',
                   edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Annual Income (k$)', color=PLOT_FG)
        ax.set_ylabel('Spending Score', color=PLOT_FG)
        ax.set_title('Customer Segments + Anomalies', color=PLOT_FG, fontweight='bold')
        ax.tick_params(colors=PLOT_FG)
        ax.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values(): spine.set_color(GRID_COL)
        ax.legend(fontsize=9, facecolor=PLOT_BG, edgecolor=GRID_COL, labelcolor=PLOT_FG)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("<div class='section-title'>Flagged Customers</div>", unsafe_allow_html=True)
        anomaly_df = df[df['Anomaly'] == -1][['Annual Income (k$)','Spending Score']].reset_index(drop=True)
        anomaly_df.index += 1
        st.dataframe(
            anomaly_df.style
            .background_gradient(cmap='Reds', subset=['Annual Income (k$)'])
            .background_gradient(cmap='Blues', subset=['Spending Score'])
            .format({'Annual Income (k$)': '{:.0f}k', 'Spending Score': '{:.0f}'}),
            use_container_width=True, height=350
        )
        st.markdown(f"""
        <div style='background:rgba(226,75,74,0.1);border:1px solid rgba(226,75,74,0.3);
        border-radius:10px;padding:1rem;margin-top:0.5rem'>
        <div style='color:#E24B4A;font-weight:600;margin-bottom:6px'>⚠️ {anomaly_count} unusual customers</div>
        <div style='color:#94a3b8;font-size:0.85rem;line-height:1.7'>
        These customers sit at the extreme edges of income or spending. 
        They don't fit cleanly into any segment.<br><br>
        <b style='color:#e2e8f0'>Recommended action:</b> Personal outreach, 
        dedicated account manager, or individual survey.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Full Dataset with Segment Labels</div>", unsafe_allow_html=True)

    display_df = df.copy()
    display_df['Segment Name'] = display_df['Segment'].map(
        {i: f"{SEG_ICONS[i]} {SEG_NAMES[i]}" for i in range(5)})
    display_df['Anomaly'] = display_df['Anomaly'].map({1:'Normal', -1:'⚠️ Anomaly'})

    show_cols = ['Annual Income (k$)','Spending Score','Segment Name','Anomaly']
    if 'Age' in display_df.columns: show_cols.insert(0,'Age')
    if 'Gender' in display_df.columns: show_cols.insert(0,'Gender')

    st.dataframe(display_df[show_cols], use_container_width=True, height=420)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = display_df[show_cols].to_csv(index=False)
        st.download_button("⬇️ Download Segmented CSV", csv,
            "shopiq_segments.csv", "text/csv", use_container_width=True)
    with col_dl2:
        st.markdown(f"""
        <div style='background:#1e2130;border:1px solid #2d3561;border-radius:10px;
        padding:0.85rem 1.25rem;text-align:center'>
        <div style='color:#64748b;font-size:0.75rem;margin-bottom:4px'>MODEL ACCURACY</div>
        <div style='color:#A3E635;font-size:1.1rem;font-weight:600'>K-Means · Inertia: {km.inertia_:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
