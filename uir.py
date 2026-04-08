import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from logic1 import load_patient_data, load_hospital_data, get_model, allocate_patients

st.set_page_config(page_title="MediCore ICU Resource Allocation", page_icon="🏥", layout="wide")

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">', unsafe_allow_html=True)

st.markdown("""
<style>
:root {
  --bg-main:       #0B1120;
  --bg-card:       #131C2E;
  --bg-card-hover: #192640;
  --bg-sidebar:    #0D1526;
  --text-primary:  #E2E8F0;
  --text-secondary:#94A3B8;
  --text-muted:    #64748B;
  --border:        rgba(255,255,255,0.06);
  --border-glow:   rgba(56,189,248,0.25);
  --cyan-500:      #22D3EE;
  --cyan-400:      #38BDF8;
  --blue-500:      #3B82F6;
  --blue-600:      #2563EB;
  --green-500:     #22C55E;
  --green-900:     rgba(34,197,94,0.12);
  --red-500:       #EF4444;
  --red-900:       rgba(239,68,68,0.12);
  --orange-500:    #F59E0B;
  --orange-900:    rgba(245,158,11,0.12);
  --purple-500:    #A78BFA;
  --purple-900:    rgba(167,139,250,0.12);
  --teal-500:      #2DD4BF;
  --teal-900:      rgba(45,212,191,0.12);
  --shadow-sm:     0 2px 8px rgba(0,0,0,0.3);
  --shadow-md:     0 4px 16px rgba(0,0,0,0.4);
  --shadow-lg:     0 8px 32px rgba(0,0,0,0.5);
  --shadow-glow-cyan: 0 0 20px rgba(34,211,238,0.15);
  --shadow-glow-blue: 0 0 20px rgba(59,130,246,0.15);
  --radius:        14px;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg-main) !important;
  color: var(--text-primary) !important;
  background-image:
    radial-gradient(ellipse at 15% 50%, rgba(34,211,238,0.04) 0%, transparent 50%),
    radial-gradient(ellipse at 85% 20%, rgba(59,130,246,0.04) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 80%, rgba(167,139,250,0.03) 0%, transparent 50%) !important;
}

/* ── Animations ── */
@keyframes fade-up {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fade-in {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.5; transform: scale(0.7); }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(34,211,238,0.1); }
  50%      { box-shadow: 0 0 20px rgba(34,211,238,0.25); }
}
@keyframes float-icon {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(-4px); }
}
@keyframes slide-in-left {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* ── Header ── */
.hospital-header {
  background: linear-gradient(135deg, #0F2447 0%, #0B1A3A 40%, #121832 100%);
  border: 1px solid rgba(56,189,248,0.15);
  border-radius: var(--radius);
  padding: 1.6rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  animation: fade-up 0.6s ease forwards;
}
.hospital-header::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan-500), var(--blue-500), var(--purple-500));
}
.hospital-header::after {
  content: '';
  position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(34,211,238,0.06), transparent 70%);
  border-radius: 50%;
}
.hospital-title {
  font-size: clamp(1.4rem, 2.5vw, 1.8rem);
  font-weight: 800;
  color: #FFFFFF;
  margin: 0;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #FFFFFF, var(--cyan-400));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hospital-subtitle {
  color: var(--text-secondary);
  font-size: 0.82rem;
  margin-top: 0.4rem;
  font-weight: 400;
}
.live-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green-500);
  animation: pulse-dot 2s ease-in-out infinite;
  box-shadow: 0 0 6px var(--green-500);
}

/* ── KPI Cards ── */
.kpi-card {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
  height: 170px;
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  animation: fade-up 0.5s ease forwards;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
.kpi-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-4px);
  box-shadow: var(--shadow-md);
}
[data-testid="stHorizontalBlock"] {
  align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > div {
  display: flex !important;
  flex-direction: column !important;
}
.kpi-card .kpi-icon-area {
  width: 40px; height: 40px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.15rem; margin-bottom: 0.4rem;
  flex-shrink: 0;
  animation: float-icon 3s ease-in-out infinite;
}
.kpi-card.cyan .kpi-icon-area    { background: rgba(34,211,238,0.1); }
.kpi-card.blue .kpi-icon-area    { background: rgba(59,130,246,0.1); }
.kpi-card.green .kpi-icon-area   { background: var(--green-900); }
.kpi-card.orange .kpi-icon-area  { background: var(--orange-900); }
.kpi-card.purple .kpi-icon-area  { background: var(--purple-900); }
.kpi-card.red .kpi-icon-area     { background: var(--red-900); }
.kpi-card.teal .kpi-icon-area    { background: var(--teal-900); }

.kpi-card::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
  opacity: 0.8;
}
.kpi-card.cyan::after   { background: linear-gradient(90deg, var(--cyan-500), transparent); }
.kpi-card.blue::after   { background: linear-gradient(90deg, var(--blue-500), transparent); }
.kpi-card.green::after  { background: linear-gradient(90deg, var(--green-500), transparent); }
.kpi-card.orange::after { background: linear-gradient(90deg, var(--orange-500), transparent); }
.kpi-card.purple::after { background: linear-gradient(90deg, var(--purple-500), transparent); }
.kpi-card.red::after    { background: linear-gradient(90deg, var(--red-500), transparent); }
.kpi-card.teal::after   { background: linear-gradient(90deg, var(--teal-500), transparent); }

.kpi-card:hover::after { opacity: 1; }

.kpi-label {
  color: var(--text-secondary);
  font-size: 0.68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-value {
  font-size: 2rem; font-weight: 800;
  color: #FFFFFF;
  line-height: 1; margin: 0.2rem 0 0.15rem;
  letter-spacing: -1px;
  white-space: nowrap;
}
.kpi-sub {
  font-size: 0.72rem; font-weight: 500;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Section Headers ── */
.section-header {
  font-size: 1rem; font-weight: 700;
  color: var(--text-primary);
  margin: 2rem 0 1rem;
  display: flex; align-items: center; gap: 10px;
  letter-spacing: -0.3px;
  animation: slide-in-left 0.4s ease forwards;
}
.section-header::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border) !important;
}
.sidebar-brand {
  text-align: center; padding: 1rem 0 0.5rem;
  animation: fade-in 0.8s ease;
}
.sidebar-brand-icon {
  font-size: 2.2rem;
  animation: float-icon 3s ease-in-out infinite;
  display: inline-block;
}
.sidebar-brand-name {
  font-size: 1.3rem; font-weight: 800;
  background: linear-gradient(135deg, var(--cyan-500), var(--blue-500));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.sidebar-brand-sub {
  font-size: 0.72rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 1.5px;
  font-weight: 600;
}

section[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(59,130,246,0.15)) !important;
  border: 1px solid rgba(34,211,238,0.3) !important;
  color: var(--cyan-500) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.5px !important;
  border-radius: 12px !important;
  padding: 0.8rem 1rem !important;
  transition: all 0.3s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(135deg, rgba(34,211,238,0.25), rgba(59,130,246,0.25)) !important;
  box-shadow: 0 0 25px rgba(34,211,238,0.3) !important;
  transform: translateY(-2px) !important;
}

/* ── Disease Buttons ── */
.stButton > button {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius) !important;
  padding: 0.7rem 1rem !important;
  font-size: 0.82rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
  white-space: normal !important;
  word-break: break-word !important;
  height: auto !important;
  min-height: 56px !important;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: var(--shadow-glow-cyan) !important;
  border-color: rgba(34,211,238,0.4) !important;
  background: var(--bg-card-hover) !important;
}

/* ── Info Boxes ── */
.info-box {
  background: var(--bg-card);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.7rem;
  display: flex; align-items: center; gap: 12px;
  transition: all 0.3s ease;
  font-size: 0.85rem;
  border: 1px solid var(--border);
  animation: fade-up 0.4s ease forwards;
}
.info-box:hover { transform: translateX(4px); box-shadow: var(--shadow-md); }
.info-box.purple { border-left: 3px solid var(--purple-500); }
.info-box.green  { border-left: 3px solid var(--green-500); }
.info-box.cyan   { border-left: 3px solid var(--cyan-500); }

/* ── Charts ── */
[data-testid="stPlotlyChart"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.5rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.3s ease !important;
  animation: fade-up 0.5s ease forwards;
}
[data-testid="stPlotlyChart"]:hover {
  box-shadow: var(--shadow-glow-blue) !important;
  border-color: rgba(59,130,246,0.2) !important;
}

/* ── Tables & Inputs ── */
[data-testid="stDataFrame"], iframe {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 2px dashed rgba(34,211,238,0.2) !important;
  border-radius: var(--radius) !important;
  transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(34,211,238,0.4) !important;
  box-shadow: var(--shadow-glow-cyan) !important;
}

hr { border-color: var(--border) !important; }

/* Download button */
[data-testid="stDownloadButton"] > button {
  background: rgba(34,197,94,0.1) !important;
  border: 1px solid rgba(34,197,94,0.3) !important;
  color: var(--green-500) !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  box-shadow: 0 0 20px rgba(34,197,94,0.2) !important;
  background: rgba(34,197,94,0.15) !important;
}

/* ── Welcome Screen ── */
.welcome-screen {
  text-align: center; padding: 5rem 2rem;
  animation: fade-up 0.8s ease forwards;
}
.welcome-icon {
  font-size: 4rem; margin-bottom: 1rem;
  animation: float-icon 3s ease-in-out infinite;
  display: inline-block;
}
.welcome-title {
  font-size: 2.2rem; font-weight: 800;
  background: linear-gradient(135deg, #FFFFFF, var(--cyan-400));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.welcome-sub {
  color: var(--text-secondary);
  font-size: 1rem; margin-top: 0.5rem;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] { color: var(--text-primary) !important; }

/* ── Stacked animation delays ── */
.kpi-card:nth-child(1) { animation-delay: 0s; }
.kpi-card:nth-child(2) { animation-delay: 0.1s; }
.kpi-card:nth-child(3) { animation-delay: 0.2s; }
.kpi-card:nth-child(4) { animation-delay: 0.3s; }
.kpi-card:nth-child(5) { animation-delay: 0.4s; }

</style>
""", unsafe_allow_html=True)

# ── Chart Config ──
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter",
    font_color="#94A3B8",
    margin=dict(l=40, r=20, t=45, b=30)
)

COLOR_STATUS = {
    "ALLOCATE":           "#22D3EE",
    "REALLOCATE":         "#F59E0B",
    "WARD_ALLOCATED":     "#A78BFA",
    "EMERGENCY_OVERFLOW": "#EF4444",
}
COLOR_SEV = {
    "Mild":     "#22C55E",
    "Moderate": "#F59E0B",
    "Critical": "#EF4444"
}

DISEASE_EMOJI = {
    "COVID-19": "🦠", "Heart Failure": "❤️‍🩹", "Stroke": "🧠",
    "Trauma": "🚑", "Sepsis": "⚠️", "Burns": "🔥",
    "Pneumonia": "🫁", "default": "🏥"
}


def get_severity_group(score):
    if score >= 0.7:   return "Critical"
    elif score >= 0.4: return "Moderate"
    else:              return "Mild"


def get_disease_emoji(disease):
    for k, v in DISEASE_EMOJI.items():
        if k.lower() in str(disease).lower():
            return v
    return DISEASE_EMOJI["default"]


def scroll_to_top():
    import streamlit.components.v1 as components
    components.html("""
        <script>
            const doc = window.parent.document;
            const main = doc.querySelector('section.main');
            if (main) { main.scrollTo({top: 0, behavior: 'instant'}); }
            window.parent.scrollTo(0, 0);
        </script>
    """, height=0)


# ─────────────────────────────────────────────────────────────────────────────
# FIX: safe_read_csv always seeks to position 0 before reading.
# This prevents EmptyDataError when the same UploadedFile object is
# passed to pd.read_csv more than once in a single Streamlit run
# (e.g. once for the sidebar preview and again for allocation).
# ─────────────────────────────────────────────────────────────────────────────
def safe_load_patients(uploaded_file):
    uploaded_file.seek(0)
    return load_patient_data(uploaded_file)


def safe_load_hospitals(uploaded_file):
    uploaded_file.seek(0)
    return load_hospital_data(uploaded_file)


def main():
    # ── Session-state defaults ──
    if 'results'          not in st.session_state: st.session_state.results          = None
    if 'page'             not in st.session_state: st.session_state.page             = 'global'
    if 'patients_cache'   not in st.session_state: st.session_state.patients_cache   = None
    if 'hospitals_cache'  not in st.session_state: st.session_state.hospitals_cache  = None
    if 'last_patient_id'  not in st.session_state: st.session_state.last_patient_id  = None
    if 'last_hospital_id' not in st.session_state: st.session_state.last_hospital_id = None

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon">🏥</div>
          <div class="sidebar-brand-name">MediCore</div>
          <div class="sidebar-brand-sub">ICU Resource Allocation</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.06); margin: 0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        st.markdown("**📂 Data Upload**")
        patient_file  = st.file_uploader("Patient Records CSV",  type=['csv'])
        hospital_file = st.file_uploader("Hospital Network CSV", type=['csv'])

        # ── Cache DataFrames in session_state so we never read a spent stream ──
        # Each file is identified by its name+size; if a new file is uploaded
        # we re-read once and cache the result.
        if hospital_file is not None:
            file_key = f"{hospital_file.name}_{hospital_file.size}"
            if st.session_state.last_hospital_id != file_key:
                hospital_file.seek(0)
                st.session_state.hospitals_cache  = load_hospital_data(hospital_file)
                st.session_state.last_hospital_id = file_key

        if patient_file is not None:
            file_key = f"{patient_file.name}_{patient_file.size}"
            if st.session_state.last_patient_id != file_key:
                patient_file.seek(0)
                st.session_state.patients_cache  = load_patient_data(patient_file)
                st.session_state.last_patient_id = file_key

        hospitals_df = st.session_state.hospitals_cache
        patients_df  = st.session_state.patients_cache

        # ── Sidebar status display ──
        if hospitals_df is not None:  
            main_hospital_name_preview = hospitals_df.iloc[0]['hospital_name']
            st.markdown(f"""
            <div class="info-box purple">
              🏨
              <div>
                <div style="font-weight:600;font-size:0.78rem;color:var(--text-muted);">Main Hospital</div>
                <div style="font-weight:700;color:var(--text-primary);">{main_hospital_name_preview}</div>
              </div>
            </div>
            <div class="info-box green">
              🤖
              <div>
                <div style="font-weight:600;font-size:0.78rem;color:var(--text-muted);">ML Model</div>
                <div style="font-weight:700;color:var(--green-500);">Logistic Regression ✓ Ready</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box cyan">
              <div>📋 Upload both CSV files<br><span style="color:var(--text-muted);font-size:0.78rem;">to initialize the system</span></div>
            </div>
            """, unsafe_allow_html=True)

        if patients_df is None or hospitals_df is None:
            st.stop()

        main_hospital_name = hospitals_df.iloc[0]['hospital_name']
        main_hospital_id   = hospitals_df.iloc[0]['hospital_id']

        model = get_model()
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("🚀 Launch Allocation Engine", use_container_width=True):
            with st.spinner("⚙️ Processing allocation matrix..."):
                results_df, final_hospitals = allocate_patients(
                    patients_df, hospitals_df, main_hospital_id, survival_model=model
                )
                st.session_state.results            = results_df
                st.session_state.final_hospitals    = final_hospitals
                st.session_state.original_hospitals = hospitals_df
                st.session_state.main_hospital      = hospitals_df.iloc[0]
                st.session_state.patients_df        = patients_df
                st.session_state.page               = 'global'
                st.rerun()

    # ── Main area: show welcome if no results yet ──
    if st.session_state.results is None:
        st.markdown("""
        <div class="welcome-screen">
          <div class="welcome-icon">🏥</div>
          <div class="welcome-title">MediCore ICU Resource Allocation</div>
          <div class="welcome-sub">Upload patient &amp; hospital data to initialize the allocation engine</div>
        </div>
        """, unsafe_allow_html=True)
        return

    results         = st.session_state.results
    final_hospitals = st.session_state.final_hospitals
    orig_hospitals  = st.session_state.original_hospitals
    patients_df     = st.session_state.patients_df

    results['severity_group'] = results['severity_score'].apply(get_severity_group)

    hosp_coords = orig_hospitals[['hospital_name', 'lat', 'lon']].drop_duplicates('hospital_name')
    results = results.merge(hosp_coords, left_on='assigned_hospital', right_on='hospital_name', how='left')
    results.rename(columns={'lat': 'hospital_lat', 'lon': 'hospital_lon'}, inplace=True)
    results['hospital_lat'] = results['hospital_lat'].fillna(0)
    results['hospital_lon'] = results['hospital_lon'].fillna(0)
    results = results.merge(patients_df[['patient_id', 'lat', 'lon']], on='patient_id', how='left')

    if st.session_state.page == 'global':
        show_global_page(results, patients_df, orig_hospitals, final_hospitals)
    else:
        show_disease_detail(
            results, patients_df,
            st.session_state.selected_disease,
            orig_hospitals, final_hospitals
        )


# ─────────────────────────────────────────────────────────────────────────────
def show_global_page(results, patients_df, orig_hospitals, final_hospitals):
# ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hospital-header">
      <div style="display:flex;justify-content:space-between;align-items:center;position:relative;z-index:1;">
        <div>
          <div class="hospital-title">🏥 MediCore ICU Resource Allocation</div>
          <div class="hospital-subtitle">Triage-based Critical Care Prioritization  ·  Real-time Allocation Engine  ·  Logistic Regression AI</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    total          = len(results)
    critical       = (results["severity_group"] == "Critical").sum()
    moderate       = (results["severity_group"] == "Moderate").sum()
    allocated      = (results["allocation_status"] == "ALLOCATE").sum()
    reallocated    = (results["allocation_status"] == "REALLOCATE").sum()
    ward_allocated = (results["allocation_status"] == "WARD_ALLOCATED").sum()
    overflow       = (results["allocation_status"] == "EMERGENCY_OVERFLOW").sum()
    total_capacity = orig_hospitals["icu_beds"].sum()
    remaining_icu  = final_hospitals["available_icu_beds"].sum()
    used_icu       = total_capacity - remaining_icu
    avg_survival   = results["survival_probability"].mean()

    # ── Row 1: 6 KPI cards ──
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="kpi-card cyan">
          <div class="kpi-icon-area">👥</div>
          <div class="kpi-label">Total Patients</div>
          <div class="kpi-value">{total}</div>
          <div class="kpi-sub" style="white-space:normal;line-height:1.5;">🔴 {critical} critical<br>🟡 {moderate} moderate</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card blue">
          <div class="kpi-icon-area">🛏️</div>
          <div class="kpi-label">ICU Allocated</div>
          <div class="kpi-value">{allocated}</div>
          <div class="kpi-sub">🔄 {reallocated} reallocated</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card purple">
          <div class="kpi-icon-area">🏨</div>
          <div class="kpi-label">Normal Ward</div>
          <div class="kpi-value">{ward_allocated}</div>
          <div class="kpi-sub">📋 General admission</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card red">
          <div class="kpi-icon-area">🚨</div>
          <div class="kpi-label">Emerg. Overflow</div>
          <div class="kpi-value">{overflow}</div>
          <div class="kpi-sub">⚠️ No bed available</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        surv_class = "green" if avg_survival > 0.7 else "orange" if avg_survival > 0.5 else "red"
        st.markdown(f"""
        <div class="kpi-card {surv_class}">
          <div class="kpi-icon-area">📊</div>
          <div class="kpi-label">Avg Survival</div>
          <div class="kpi-value">{avg_survival:.2f}</div>
          <div class="kpi-sub">🧬 Probability index</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        pct = (used_icu / total_capacity * 100) if total_capacity > 0 else 0
        pct_color = "#EF4444" if pct > 85 else "#F59E0B" if pct > 60 else "#22D3EE"
        st.markdown(f"""
        <div class="kpi-card red">
          <div class="kpi-icon-area">📈</div>
          <div class="kpi-label">ICU Utilization</div>
          <div class="kpi-value" style="color:{pct_color};">{pct:.0f}%</div>
          <div class="kpi-sub">🛏️ {used_icu} / {total_capacity} beds used</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Row 2: 2 charts ──
    dc1, dc2 = st.columns(2)

    with dc1:
        fig1 = px.histogram(results, x="severity_score", nbins=25,
                            color="severity_group", color_discrete_map=COLOR_SEV,
                            title="📊 Severity Distribution")
        fig1.update_layout(**CHART_LAYOUT, height=300, bargap=0.05)
        fig1.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
        fig1.update_xaxes(showgrid=False)
        st.plotly_chart(fig1, use_container_width=True, theme=None, config={'displayModeBar': False})

    with dc2:
        fig2 = px.histogram(results, x="survival_probability", nbins=25,
                            color="allocation_status", color_discrete_map=COLOR_STATUS,
                            barmode="stack", title="💓 Survival by Allocation")
        fig2.update_layout(**CHART_LAYOUT, height=300, bargap=0.05)
        fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
        fig2.update_xaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True, theme=None, config={'displayModeBar': False})

    st.divider()

    # ── Disease filter buttons ──
    st.markdown('<div class="section-header">🦠 Filter by Condition</div>', unsafe_allow_html=True)
    diseases = results['disease'].unique()
    cols = st.columns(min(len(diseases), 4))
    for i, disease in enumerate(diseases):
        emoji = get_disease_emoji(disease)
        count = len(results[results['disease'] == disease])
        crit_count = (results[(results['disease'] == disease) & (results['severity_group'] == 'Critical')]).shape[0]
        with cols[i % len(cols)]:
            if st.button(
                f"{emoji} {disease}\n🔴 {crit_count} Critical · {count} Total",
                key=f"card_{disease}", use_container_width=True
            ):
                st.session_state.page = 'disease_detail'
                st.session_state.selected_disease = disease
                st.rerun()

    st.divider()

    # ── Hospital Bed Availability Before vs After ──
    st.markdown('<div class="section-header">🏥 Hospital Network — ICU Beds Before vs After</div>', unsafe_allow_html=True)
    hosp_compare = pd.DataFrame({
        "Hospital":   orig_hospitals["hospital_name"].values,
        "ICU Before": orig_hospitals["available_icu_beds"].values,
        "ICU After":  final_hospitals["available_icu_beds"].values,
    })
    fig_beds = go.Figure()
    fig_beds.add_trace(go.Bar(name="Before", x=hosp_compare["Hospital"], y=hosp_compare["ICU Before"],
                              marker_color="#22D3EE", marker_line_width=0))
    fig_beds.add_trace(go.Bar(name="After",  x=hosp_compare["Hospital"], y=hosp_compare["ICU After"],
                              marker_color="#EF4444", marker_line_width=0))
    fig_beds.update_layout(**CHART_LAYOUT, barmode="group", xaxis_tickangle=-30, height=350,
                           title="ICU Bed Availability Before vs After Allocation")
    st.plotly_chart(fig_beds, use_container_width=True, theme=None)

    # ── Resource Utilisation Overview ──
    st.markdown('<div class="section-header">🩻 Resource Utilization Overview</div>', unsafe_allow_html=True)

    total_vents_before  = orig_hospitals["ventilators"].sum() if "ventilators" in orig_hospitals.columns else orig_hospitals.get("available_ventilators", pd.Series([0])).sum()
    total_vents_after   = final_hospitals["available_ventilators"].sum() if "available_ventilators" in final_hospitals.columns else 0
    vents_used          = max(total_vents_before - total_vents_after, 0)

    total_oxygen_before = orig_hospitals["oxygen_supply"].sum() if "oxygen_supply" in orig_hospitals.columns else 0
    total_oxygen_after  = final_hospitals["oxygen_supply"].sum() if "oxygen_supply" in final_hospitals.columns else 0
    oxygen_consumed     = max(total_oxygen_before - total_oxygen_after, 0)

    total_doctors = orig_hospitals["doctors_count"].sum() if "doctors_count" in orig_hospitals.columns else 0
    total_nurses  = orig_hospitals["nurses_count"].sum()  if "nurses_count"  in orig_hospitals.columns else 0

    rv1, rv2, rv3 = st.columns(3)
    with rv1:
        vent_pct = (vents_used / total_vents_before * 100) if total_vents_before > 0 else 0
        vc = "red" if vent_pct > 85 else "orange" if vent_pct > 60 else "cyan"
        st.markdown(f"""
        <div class="kpi-card {vc}">
          <div class="kpi-icon-area">🫁</div>
          <div class="kpi-label">Ventilators Used</div>
          <div class="kpi-value">{vents_used}/{total_vents_before}</div>
          <div class="kpi-sub">📊 {vent_pct:.1f}% utilization</div>
        </div>""", unsafe_allow_html=True)
    with rv2:
        oxy_pct = (oxygen_consumed / total_oxygen_before * 100) if total_oxygen_before > 0 else 0
        oc = "red" if oxy_pct > 85 else "orange" if oxy_pct > 60 else "teal"
        st.markdown(f"""
        <div class="kpi-card {oc}">
          <div class="kpi-icon-area">💨</div>
          <div class="kpi-label">Oxygen Consumed</div>
          <div class="kpi-value">{oxygen_consumed}/{total_oxygen_before}</div>
          <div class="kpi-sub">📊 {oxy_pct:.1f}% consumed</div>
        </div>""", unsafe_allow_html=True)
    with rv3:
        st.markdown(f"""
        <div class="kpi-card green">
          <div class="kpi-icon-area">👨‍⚕️</div>
          <div class="kpi-label">Staff Available</div>
          <div class="kpi-value">{total_doctors + total_nurses}</div>
          <div class="kpi-sub">🩺 {total_doctors} doctors · 👩‍⚕️ {total_nurses} nurses</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Ventilator chart
    if "available_ventilators" in orig_hospitals.columns and "available_ventilators" in final_hospitals.columns:
        hosp_vent = pd.DataFrame({
            "Hospital":            orig_hospitals["hospital_name"].values,
            "Ventilators Before":  orig_hospitals["available_ventilators"].values,
            "Ventilators After":   final_hospitals["available_ventilators"].values,
        })
        # fig_vent = go.Figure()
        # fig_vent.add_trace(go.Bar(name="Before", x=hosp_vent["Hospital"], y=hosp_vent["Ventilators Before"], marker_color="#22D3EE"))
        # fig_vent.add_trace(go.Bar(name="After",  x=hosp_vent["Hospital"], y=hosp_vent["Ventilators After"],  marker_color="#F59E0B"))
        # fig_vent.update_layout(**CHART_LAYOUT, barmode="group", xaxis_tickangle=-30, height=320,
        #                        title="Ventilator Availability Before vs After")
        # st.plotly_chart(fig_vent, use_container_width=True, theme=None)

    # Staff chart
    # if "doctors_count" in orig_hospitals.columns and "nurses_count" in orig_hospitals.columns:
    #     hosp_staff = orig_hospitals[["hospital_name", "doctors_count", "nurses_count"]].copy()
    #     fig_staff = px.bar(hosp_staff, x="hospital_name", y=["doctors_count", "nurses_count"],
    #                        barmode="group", title="Medical Staff per Hospital",
    #                        color_discrete_map={"doctors_count": "#A78BFA", "nurses_count": "#22C55E"},
    #                        labels={"value": "Count", "variable": "Role", "hospital_name": "Hospital"})
    #     fig_staff.update_layout(**CHART_LAYOUT, xaxis_tickangle=-30, height=320)
    #     st.plotly_chart(fig_staff, use_container_width=True, theme=None)

    # st.divider()

    # Ethical score scatter
      # if 'ethical_score' in results.columns:
      #     st.markdown('<div class="section-header">⚖️ Ethical Score vs Severity</div>', unsafe_allow_html=True)
      #     fig_eth = px.scatter(
      #         results,
      #         x="severity_score", y="ethical_score",
      #         color="severity_group",
      #         symbol="allocation_status",
      #         size="survival_probability",
      #         color_discrete_map=COLOR_SEV,
      #         symbol_map={"ALLOCATE": "circle", "REALLOCATE": "square",
      #                     "WARD_ALLOCATED": "diamond", "EMERGENCY_OVERFLOW": "x"},
      #         hover_data=["patient_id", "disease", "allocation_status"],
      #         title="Clustering Groups (Mild / Moderate / Critical) vs Ethical Score"
      #     )
      #     fig_eth.update_layout(**CHART_LAYOUT, height=450)
      #     st.plotly_chart(fig_eth, use_container_width=True, theme=None)

    # Hospital network table
    st.markdown('<div class="section-header">🏨 Hospital Network Status</div>', unsafe_allow_html=True)
    hosp_display = final_hospitals.copy()
    if all(c in hosp_display.columns for c in ['hospital_name', 'icu_beds', 'available_icu_beds']):
        hosp_display = hosp_display[['hospital_name', 'icu_beds', 'available_icu_beds']].copy()
        hosp_display['Used Beds']     = hosp_display['icu_beds'] - hosp_display['available_icu_beds']
        hosp_display['Utilization %'] = ((hosp_display['Used Beds'] / hosp_display['icu_beds']) * 100).round(1)
    st.dataframe(hosp_display, use_container_width=True, height=200)

    st.divider()

    # ── Map ──
    st.markdown('<div class="section-header">🗺️ Geographic Patient Mapping</div>', unsafe_allow_html=True)
    m = folium.Map(location=[28.6, 77.2], zoom_start=11, tiles="CartoDB dark_matter")
    for _, h in orig_hospitals.iterrows():
        folium.Marker(
            [h["lat"], h["lon"]],
            popup=f"🏥 {h['hospital_name']}<br>🛏️ ICU: {h.get('available_icu_beds', 'N/A')}",
            icon=folium.Icon(color="darkblue", icon="plus-sign", prefix="glyphicon")
        ).add_to(m)

    color_map = {"ALLOCATE": "#22D3EE", "REALLOCATE": "#F59E0B",
                 "WARD_ALLOCATED": "#A78BFA", "EMERGENCY_OVERFLOW": "#EF4444"}
    for _, r in results.iterrows():
        if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=5,
                color=color_map.get(r["allocation_status"], "gray"),
                fill=True, fill_opacity=0.7,
                popup=(
                    f"🆔 {r['patient_id']}<br>🦠 {r['disease']}<br>⚠️ {r['allocation_status']}"
                    + (f"<br>📍 {r['distance_km']:.1f} km" if 'distance_km' in r and pd.notna(r.get('distance_km')) else "")
                )
            ).add_to(m)
        if r.get("hospital_lat") and r.get("hospital_lon") and r.get("allocation_status") != "EMERGENCY_OVERFLOW":
            folium.PolyLine(
                [[r["lat"], r["lon"]], [r["hospital_lat"], r["hospital_lon"]]],
                color=color_map.get(r["allocation_status"], "gray"), weight=1.5, opacity=0.25
            ).add_to(m)
    st_folium(m, width=1200, height=450, use_container_width=True)

    st.divider()

    # ── Patient Roster ──
    colA, colB = st.columns([3, 1])
    with colA:
        st.markdown('<div class="section-header">📋 Patient Allocation Roster</div>', unsafe_allow_html=True)
    with colB:
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Results CSV", data=csv,
                           file_name="allocation_results.csv", mime="text/csv",
                           use_container_width=True)

    status_filter = st.multiselect(
        "🔍 Filter by Allocation Status",
        ["ALLOCATE", "REALLOCATE", "WARD_ALLOCATED", "EMERGENCY_OVERFLOW"],
        default=["ALLOCATE", "REALLOCATE", "WARD_ALLOCATED", "EMERGENCY_OVERFLOW"]
    )
    filtered = results[results["allocation_status"].isin(status_filter)]

    display_cols = ['patient_id', 'disease', 'age', 'severity_score', 'survival_probability',
                    'assigned_hospital', 'allocation_status', 'reason']
    if 'distance_km' in filtered.columns:
        display_cols = ['patient_id', 'disease', 'age', 'severity_score', 'survival_probability',
                        'assigned_hospital', 'distance_km', 'allocation_status', 'reason']
    if 'ethical_score' in filtered.columns:
        display_cols.append('ethical_score')

    st.dataframe(filtered[display_cols], use_container_width=True, height=500)


# ─────────────────────────────────────────────────────────────────────────────
def show_disease_detail(results, patients_df, disease, orig_hospitals, final_hospitals):
# ─────────────────────────────────────────────────────────────────────────────
    scroll_to_top()
    emoji = get_disease_emoji(disease)
    st.markdown(f"""
    <div class="hospital-header">
      <div class="hospital-title">{emoji} {disease} — Deep Analysis</div>
      <div class="hospital-subtitle">Disease-level patient breakdown  ·  Severity &amp; survival metrics</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Dashboard", use_container_width=False):
        st.session_state.page = 'global'
        st.rerun()

    df = results[results['disease'] == disease].copy()
    if df.empty:
        st.warning(f"⚠️ No patients found for {disease}")
        return

    total       = len(df)
    critical    = (df["severity_group"] == "Critical").sum()
    allocated   = (df["allocation_status"] == "ALLOCATE").sum()
    reallocated = (df["allocation_status"] == "REALLOCATE").sum()
    ward        = (df["allocation_status"] == "WARD_ALLOCATED").sum()
    overflow    = (df["allocation_status"] == "EMERGENCY_OVERFLOW").sum()
    avg_sev     = df["severity_score"].mean()
    avg_surv    = df["survival_probability"].mean()
    avg_dist    = df["distance_km"].mean() if "distance_km" in df.columns else None

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""<div class="kpi-card cyan">
          <div class="kpi-icon-area">👥</div>
          <div class="kpi-label">Patients</div>
          <div class="kpi-value">{total}</div>
          <div class="kpi-sub">🔴 {critical} critical</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card blue">
          <div class="kpi-icon-area">🛏️</div>
          <div class="kpi-label">ICU Beds</div>
          <div class="kpi-value">{allocated}</div>
          <div class="kpi-sub">🔄 {reallocated} reallocated</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card purple">
          <div class="kpi-icon-area">🏨</div>
          <div class="kpi-label">Ward</div>
          <div class="kpi-value">{ward}</div>
          <div class="kpi-sub">📋 General</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card red">
          <div class="kpi-icon-area">🚨</div>
          <div class="kpi-label">Overflow</div>
          <div class="kpi-value">{overflow}</div>
          <div class="kpi-sub">⚠️ No bed</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        sev_class = "red" if avg_sev > 0.7 else "orange" if avg_sev > 0.4 else "green"
        st.markdown(f"""<div class="kpi-card {sev_class}">
          <div class="kpi-icon-area">⚠️</div>
          <div class="kpi-label">Avg Severity</div>
          <div class="kpi-value">{avg_sev:.3f}</div>
          <div class="kpi-sub">📈 Severity index</div>
        </div>""", unsafe_allow_html=True)
    with col6:
        dist_val = f"{avg_dist:.1f}" if avg_dist is not None else "N/A"
        st.markdown(f"""<div class="kpi-card teal">
          <div class="kpi-icon-area">📍</div>
          <div class="kpi-label">Avg Distance</div>
          <div class="kpi-value">{dist_val}</div>
          <div class="kpi-sub">🚑 km to hospital</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Resource usage for this disease ──
    st.markdown(f'<div class="section-header">🩻 Resource Usage — {disease}</div>', unsafe_allow_html=True)

    df_res = df.merge(patients_df[['patient_id', 'spo2']], on='patient_id', how='left') \
        if 'spo2' in patients_df.columns else df.copy()

    if 'spo2' in df_res.columns:
        df_res['needs_ventilator'] = (df_res['spo2'] < 90) | (df_res['disease'].str.lower() == 'covid-19')
    else:
        df_res['needs_ventilator'] = False

    df_res['oxygen_consumed'] = 0.0
    df_res.loc[df_res['allocation_status'].isin(['ALLOCATE', 'REALLOCATE']), 'oxygen_consumed'] = 1.0
    df_res.loc[df_res['allocation_status'] == 'WARD_ALLOCATED', 'oxygen_consumed'] = 0.5
    df_res['ventilator_used'] = (
        df_res['needs_ventilator'] & df_res['allocation_status'].isin(['ALLOCATE', 'REALLOCATE'])
    )

    vents_needed = int(df_res['needs_ventilator'].sum())
    vents_used   = int(df_res['ventilator_used'].sum())
    oxygen_total = df_res['oxygen_consumed'].sum()

    rr1, rr2, rr3 = st.columns(3)
    with rr1:
        st.markdown(f"""<div class="kpi-card cyan">
          <div class="kpi-icon-area">🫁</div>
          <div class="kpi-label">Ventilators Needed</div>
          <div class="kpi-value">{vents_needed}</div>
          <div class="kpi-sub">✅ {vents_used} in use (ICU)</div>
        </div>""", unsafe_allow_html=True)
    with rr2:
        st.markdown(f"""<div class="kpi-card teal">
          <div class="kpi-icon-area">💨</div>
          <div class="kpi-label">Oxygen Consumed</div>
          <div class="kpi-value">{oxygen_total:.1f}</div>
          <div class="kpi-sub">📦 units total</div>
        </div>""", unsafe_allow_html=True)
    with rr3:
        st.markdown(f"""<div class="kpi-card blue">
          <div class="kpi-icon-area">🛏️</div>
          <div class="kpi-label">ICU Patients</div>
          <div class="kpi-value">{allocated + reallocated}</div>
          <div class="kpi-sub">ICU allocated + reallocated</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    status_break = df_res.groupby('allocation_status').agg(
        count=('patient_id', 'count'),
        vents_needed=('needs_ventilator', 'sum'),
        oxygen_used=('oxygen_consumed', 'sum')
    ).reset_index()
    fig_res = px.bar(
        status_break, x='allocation_status',
        y=['count', 'vents_needed', 'oxygen_used'],
        barmode='group',
        title=f"Resource Demand by Allocation Status — {disease}",
        color_discrete_map={"count": "#22D3EE", "vents_needed": "#A78BFA", "oxygen_used": "#2DD4BF"},
        labels={'value': 'Count / Units', 'variable': 'Resource', 'allocation_status': 'Status'}
    )
    fig_res.update_layout(**CHART_LAYOUT, height=350)
    st.plotly_chart(fig_res, use_container_width=True, theme=None)

    st.divider()

    # ── Analytics charts ──
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="severity_score", nbins=20,
                           color="severity_group", color_discrete_map=COLOR_SEV,
                           title=f"📊 Severity – {disease}")
        fig.update_layout(**CHART_LAYOUT, height=320)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with c2:
        fig = px.histogram(df, x="survival_probability", nbins=15,
                           color="allocation_status", color_discrete_map=COLOR_STATUS,
                           barmode="stack", title=f"💓 Survival – {disease}")
        fig.update_layout(**CHART_LAYOUT, height=320)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # if 'ethical_score' in df.columns:
    #     fig_eth = px.scatter(
    #         df, x="severity_score", y="ethical_score",
    #         color="allocation_status", size="survival_probability",
    #         color_discrete_map=COLOR_STATUS,
    #         hover_data=["patient_id"],
    #         title=f"⚖️ Ethical Score vs Severity — {disease}"
    #     )
    #     fig_eth.update_layout(**CHART_LAYOUT, height=400)
    #     st.plotly_chart(fig_eth, use_container_width=True, theme=None)

    # ── Geographic view ──
    st.markdown(f'<div class="section-header">🗺️ Geographic View — {disease}</div>', unsafe_allow_html=True)
    m = folium.Map(location=[28.6, 77.2], zoom_start=11, tiles="CartoDB dark_matter")
    for _, h in orig_hospitals.iterrows():
        folium.Marker(
            [h["lat"], h["lon"]], popup=f"🏥 {h['hospital_name']}",
            icon=folium.Icon(color="darkblue", icon="plus-sign", prefix="glyphicon")
        ).add_to(m)
    color_map = {"ALLOCATE": "#22D3EE", "REALLOCATE": "#F59E0B",
                 "WARD_ALLOCATED": "#A78BFA", "EMERGENCY_OVERFLOW": "#EF4444"}
    for _, r in df.iterrows():
        if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=5,
                color=color_map.get(r["allocation_status"], "gray"),
                fill=True, fill_opacity=0.7,
                popup=f"🆔 {r['patient_id']}<br>⚠️ {r['allocation_status']}"
            ).add_to(m)
        if r.get("hospital_lat") and r.get("hospital_lon") and r.get("allocation_status") != "EMERGENCY_OVERFLOW":
            folium.PolyLine(
                [[r["lat"], r["lon"]], [r["hospital_lat"], r["hospital_lon"]]],
                color=color_map.get(r["allocation_status"], "gray"), weight=1.5, opacity=0.25
            ).add_to(m)
    st_folium(m, width=1200, height=400, use_container_width=True)

    # ── Patient Roster ──
    colA, colB = st.columns([3, 1])
    with colA:
        st.markdown(f'<div class="section-header">{emoji} Patient Roster — {disease}</div>', unsafe_allow_html=True)
    with colB:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Detail CSV", data=csv,
                           file_name=f"{disease}_results.csv", mime="text/csv",
                           use_container_width=True)

    status_filter = st.multiselect(
        "🔍 Filter by Allocation Status",
        ["ALLOCATE", "REALLOCATE", "WARD_ALLOCATED", "EMERGENCY_OVERFLOW"],
        default=["ALLOCATE", "REALLOCATE", "WARD_ALLOCATED", "EMERGENCY_OVERFLOW"],
        key="disease_status_filter"
    )
    search = st.text_input("🔎 Search Patient ID or Hospital", key="disease_search")

    filtered = df[df["allocation_status"].isin(status_filter)]
    if search:
        filtered = filtered[
            filtered['patient_id'].astype(str).str.contains(search, case=False) |
            filtered['assigned_hospital'].astype(str).str.contains(search, case=False)
        ]

    display_cols = ['patient_id', 'age', 'severity_score', 'survival_probability',

                    'assigned_hospital', 'allocation_status', 'reason']
    if 'distance_km' in filtered.columns:
        display_cols = ['patient_id', 'age', 'severity_score', 'survival_probability',
                        'assigned_hospital', 'distance_km', 'allocation_status', 'reason']
    if 'ethical_score' in filtered.columns:
        display_cols.append('ethical_score')

    st.dataframe(filtered[display_cols], use_container_width=True, height=500)


if __name__ == "__main__":
    main()