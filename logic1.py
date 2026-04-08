# ============================================
# LOGIC1.PY - ICU Allocation Backend (Fixed)
# Triage-based: only critical patients go to ICU
# Severity groups by KMeans clustering
# Logistic Regression survival classifier (pre-trained on synthetic data)
# Dynamic resource management: oxygen + ventilators
# ============================================

import pandas as pd
import numpy as np
import os
import pickle
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import streamlit as st

# -------------------------------------------------------------------
# Data Loading – preserves hospital_id exactly as in CSV
# -------------------------------------------------------------------
def load_patient_data(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a patient CSV file.")
        st.stop()
    return pd.read_csv(uploaded_file)

def load_hospital_data(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a hospital CSV file.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    
    required_cols = ['hospital_name', 'lat', 'lon', 'available_beds', 'icu_beds',
                     'available_icu_beds', 'ventilators', 'available_ventilators',
                     'specializations', 'doctors_count', 'nurses_count', 'oxygen_supply']
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"Missing column '{col}' in hospital CSV. Adding default values.")
            if col in ['doctors_count', 'nurses_count']:
                df[col] = 50
            elif col == 'oxygen_supply':
                df[col] = 200
            else:
                df[col] = 0
    return df

# -------------------------------------------------------------------
# Pre-trained Logistic Regression Model (trained on synthetic data)
# -------------------------------------------------------------------
MODEL_PATH = "logistic_model.pkl"

def generate_synthetic_training_data(n_samples=20000, random_state=42):
    """Generate the improved synthetic dataset used for training the logistic regression model."""
    np.random.seed(random_state)
    
    age = np.random.randint(18, 95, n_samples)
    comorbidity_score = np.random.uniform(0, 10, n_samples)
    heart_rate_mean = np.random.normal(80, 20, n_samples)
    systolic_bp_mean = np.random.normal(120, 15, n_samples)
    lactate_mean = np.random.gamma(2, 1, n_samples)
    spo2_mean = np.random.normal(95, 5, n_samples)
    ventilation_required = np.random.binomial(1, 0.3, n_samples)
    vasopressor_used = np.random.binomial(1, 0.2, n_samples)
    apache_score = np.random.normal(20, 10, n_samples)
    sofa_score = np.random.normal(5, 3, n_samples)
    
    shock_index = heart_rate_mean / systolic_bp_mean
    age_comorbidity = age * comorbidity_score / 100
    lactate_high = (lactate_mean > 2).astype(int)
    hypoxia = (spo2_mean < 90).astype(int)
    tachycardia = (heart_rate_mean > 100).astype(int)
    hypotension = (systolic_bp_mean < 90).astype(int)
    
    log_odds = (
        -2.5
        + 3.0 * lactate_high
        + 2.5 * (shock_index > 0.9).astype(int)
        + 2.0 * (age_comorbidity > 15).astype(int)
        + 2.0 * hypoxia
        + 1.5 * ventilation_required
        + 1.2 * vasopressor_used
        + 1.0 * (apache_score > 25).astype(int)
        + 1.0 * (sofa_score > 8).astype(int)
        + 1.0 * tachycardia
        + 1.0 * hypotension
    )
    prob_mortality = 1 / (1 + np.exp(-log_odds))
    mortality_label = (np.random.rand(n_samples) < prob_mortality).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'comorbidity_score': comorbidity_score,
        'heart_rate_mean': heart_rate_mean,
        'systolic_bp_mean': systolic_bp_mean,
        'lactate_mean': lactate_mean,
        'spo2_mean': spo2_mean,
        'ventilation_required': ventilation_required,
        'vasopressor_used': vasopressor_used,
        'apache_score': apache_score,
        'sofa_score': sofa_score,
        'shock_index': shock_index,
        'age_comorbidity': age_comorbidity,
        'lactate_high': lactate_high,
        'hypoxia': hypoxia,
        'tachycardia': tachycardia,
        'hypotension': hypotension,
        'mortality_label': mortality_label
    })
    return df

def train_and_save_logistic_model():
    """Train logistic regression on synthetic data and save to disk."""
    st.info("Training logistic regression model on synthetic data (high performance). This will run once.")
    df_syn = generate_synthetic_training_data()
    X = df_syn.drop('mortality_label', axis=1)
    y = df_syn['mortality_label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression with balanced class weights
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model, scaler, and feature names
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }, f)
    st.success(f"Model saved to {MODEL_PATH}")
    return model, scaler, list(X.columns)

@st.cache_resource
def load_logistic_model():
    """Load pre-trained logistic regression model, scaler, and feature names (cached)."""
    if not os.path.exists(MODEL_PATH):
        st.warning("Pre-trained model not found. Training now...")
        model, scaler, feature_names = train_and_save_logistic_model()
    else:
        with open(MODEL_PATH, 'rb') as f:
            obj = pickle.load(f)
        model = obj['model']
        scaler = obj['scaler']
        feature_names = obj['feature_names']
    return model, scaler, feature_names

def get_survival_model():
    """Return the logistic regression model (for compatibility)."""
    model, _, _ = load_logistic_model()
    return model

def get_model():
    """Alias for get_survival_model – required by ui2.py."""
    return get_survival_model()

def predict_survival(model, patient):
    """
    Predict survival probability (1 - mortality) using logistic regression.
    patient: dictionary with keys: age, spo2, bp, heart_rate, disease, comorbidities, etc.
    """
    # Load model, scaler, and feature names from cache
    _, scaler, feature_names = load_logistic_model()
    model, _, _ = load_logistic_model()  # get the actual model (could reuse but okay)
    
    # Extract features needed by the logistic regression model
    bp_sys = extract_systolic_bp(patient['bp'])
    hr = patient['heart_rate']
    spo2 = patient['spo2']
    age = patient['age']
    
    # Compute synthetic features (same as in training)
    comorbidity_score = calculate_comorbidity_score(patient['comorbidities'])
    lactate_mean = 1.5  # default; in real data you would have this value
    shock_index = hr / bp_sys
    age_comorbidity = age * comorbidity_score / 100
    lactate_high = 1 if lactate_mean > 2 else 0
    hypoxia = 1 if spo2 < 90 else 0
    tachycardia = 1 if hr > 100 else 0
    hypotension = 1 if bp_sys < 90 else 0
    ventilation_required = 1 if patient.get('ventilation_required', 0) else 0
    vasopressor_used = 1 if patient.get('vasopressor_used', 0) else 0
    apache_score = 20   # default; not available in patient data
    sofa_score = 5      # default
    
    # Build feature vector in the same order as training
    features_dict = {
        'age': age,
        'comorbidity_score': comorbidity_score,
        'heart_rate_mean': hr,
        'systolic_bp_mean': bp_sys,
        'lactate_mean': lactate_mean,
        'spo2_mean': spo2,
        'ventilation_required': ventilation_required,
        'vasopressor_used': vasopressor_used,
        'apache_score': apache_score,
        'sofa_score': sofa_score,
        'shock_index': shock_index,
        'age_comorbidity': age_comorbidity,
        'lactate_high': lactate_high,
        'hypoxia': hypoxia,
        'tachycardia': tachycardia,
        'hypotension': hypotension
    }
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features_dict])[feature_names]
    X_scaled = scaler.transform(X)
    mortality_prob = model.predict_proba(X_scaled)[0][1]
    return 1 - mortality_prob

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def extract_systolic_bp(bp_str):
    try:
        return int(str(bp_str).split('/')[0])
    except:
        return 120

def calculate_comorbidity_score(comorbidities):
    mapping = {'none':0, 'asthma':3, 'diabetes':4, 'hypertension':4, 'obesity':3, 'copd':5}
    return mapping.get(comorbidities, 3)

def get_patient_features_for_clustering(patient):
    return np.array([
        patient['spo2'],
        extract_systolic_bp(patient['bp']),
        patient['heart_rate'],
        patient['age'],
        calculate_comorbidity_score(patient['comorbidities'])
    ])

@st.cache_resource
def train_clustering_model(patients_df):
    features = []
    for _, patient in patients_df.iterrows():
        features.append(get_patient_features_for_clustering(patient))
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    cluster_means = []
    for i in range(3):
        mask = kmeans.labels_ == i
        if np.any(mask):
            severities = [calculate_severity(patients_df.iloc[j]) for j in np.where(mask)[0]]
            cluster_means.append((i, np.mean(severities)))
        else:
            cluster_means.append((i, -1))
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    cluster_to_group = {
        cluster_means[0][0]: 'Critical',
        cluster_means[1][0]: 'Moderate',
        cluster_means[2][0]: 'Mild'
    }
    return kmeans, scaler, cluster_to_group

def assign_severity_group(patient, kmeans, scaler, cluster_to_group):
    features = get_patient_features_for_clustering(patient).reshape(1, -1)
    features_scaled = scaler.transform(features)
    cluster = kmeans.predict(features_scaled)[0]
    return cluster_to_group[cluster]

def calculate_severity(patient):
    severity = 0
    if patient['spo2'] < 85: severity += 40
    elif patient['spo2'] < 90: severity += 30
    elif patient['spo2'] < 94: severity += 20
    elif patient['spo2'] < 96: severity += 10
    bp_sys = extract_systolic_bp(patient['bp'])
    if bp_sys < 90 or bp_sys > 160: severity += 20
    elif bp_sys < 100 or bp_sys > 150: severity += 10
    hr = patient['heart_rate']
    if hr < 60 or hr > 120: severity += 20
    elif hr < 70 or hr > 110: severity += 10
    comorbidity_score = calculate_comorbidity_score(patient['comorbidities'])
    severity += comorbidity_score * 5
    age = patient['age']
    if age > 80: severity += 10
    elif age > 70: severity += 7
    elif age > 60: severity += 5
    elif age > 50: severity += 3
    return min(severity / 100, 1.0)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    return R * 2 * atan2(sqrt(sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2),
                         sqrt(1 - (sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2)))

# -------------------------------------------------------------------
# Dynamic Resource Functions (Oxygen + Ventilators)
# -------------------------------------------------------------------
def needs_ventilator(patient):
    return (patient['spo2'] < 90) or (patient['disease'] == 'COVID-19')

def patient_needs_oxygen(patient):
    return (patient['spo2'] < 90) or (patient['disease'] == 'COVID-19')

def oxygen_consumption_for_icu(patient, emergency_mode=False):
    return 1.0

def oxygen_consumption_for_ward(patient):
    return 1.0 if patient_needs_oxygen(patient) else 0.0

def can_allocate_icu(hospital, patient, emergency_mode=False):
    if effective_icu_beds(hospital, emergency_mode) <= 0:
        return False
    if needs_ventilator(patient):
        if hospital['available_ventilators'] <= 0:
            return False
    if hospital['oxygen_supply'] - 1 < 0:
        return False
    return True

def effective_icu_beds(hospital, emergency_mode=False):
    physical = hospital['available_icu_beds']
    nurses = hospital['nurses_count']
    doctors = hospital['doctors_count']
    vents = hospital['available_ventilators']
    oxygen_units = hospital['oxygen_supply']
    
    if emergency_mode:
        max_by_nurses = max(1, nurses // 4) if nurses >= 4 else nurses
        max_by_doctors = max(1, doctors // 8) if doctors >= 8 else doctors
    else:
        max_by_nurses = nurses // 2
        max_by_doctors = doctors // 4
    
    max_by_vents = vents
    oxygen_capacity = int(oxygen_units / 1.0)
    effective = min(physical, max_by_nurses, max_by_doctors, max_by_vents, oxygen_capacity)
    
    if emergency_mode and physical > 0 and (nurses > 0 or doctors > 0) and effective == 0:
        effective = 1
    return max(0, effective)

def hospital_capacity_score(hospital, emergency_mode=False):
    eff_icu = effective_icu_beds(hospital, emergency_mode)
    icu_ratio = eff_icu / max(hospital['icu_beds'], 1)
    vent_ratio = hospital['available_ventilators'] / max(hospital['ventilators'], 1)
    oxygen_ratio = hospital['oxygen_supply'] / max(hospital['oxygen_supply'], 200)
    oxygen_score = min(1.0, oxygen_ratio)
    if emergency_mode:
        staff_score = min(1.0, (hospital['doctors_count'] / 20) * (hospital['nurses_count'] / 40))
    else:
        staff_score = min(1.0, (hospital['doctors_count'] / 50) * (hospital['nurses_count'] / 100))
    return 0.3 * icu_ratio + 0.3 * vent_ratio + 0.2 * oxygen_score + 0.2 * staff_score

def is_emergency_mode(patients_df, hospitals_df, critical_unallocated_count=0):
    total_icu_capacity = sum(h['icu_beds'] for _, h in hospitals_df.iterrows())
    total_icu_used = sum(h['icu_beds'] - h['available_icu_beds'] for _, h in hospitals_df.iterrows())
    occupancy_rate = total_icu_used / total_icu_capacity if total_icu_capacity > 0 else 0
    return (occupancy_rate > 0.85) or (critical_unallocated_count > 0)

# -------------------------------------------------------------------
# Main Allocation Function
# -------------------------------------------------------------------
def allocate_patients(patients_df, hospitals_df, main_hospital_id, survival_model=None):
    resources = hospitals_df.copy()
    patients = patients_df.copy().reset_index(drop=True)

    # Use pre-trained logistic regression model (ignores survival_model argument)
    survival_model = get_survival_model()   # not used directly, predict_survival uses the model

    kmeans, scaler, cluster_to_group = train_clustering_model(patients)

    patient_info = []
    for idx, patient in patients.iterrows():
        severity_score = calculate_severity(patient)
        group = assign_severity_group(patient, kmeans, scaler, cluster_to_group)
        surv = predict_survival(survival_model, patient)   # uses logistic regression
        patient_info.append({
            'idx': idx,
            'patient': patient,
            'severity': severity_score,
            'group': group,
            'survival': surv
        })

    critical = [p for p in patient_info if p['group'] == 'Critical']
    non_critical = [p for p in patient_info if p['group'] != 'Critical']

    emergency = is_emergency_mode(patients_df, resources, len(critical))

    # Ethical scoring for critical patients
    for p in critical:
        disease = p['patient']['disease']
        base_eth = 0.4 * p['severity'] + 0.3 * p['survival'] + 0.2 * 0.8 + 0.1 * 0.5
        if disease == 'COVID-19':
            base_eth += 0.15
        elif disease == 'Cardiovascular Disease':
            base_eth += 0.1
        p['eth_score'] = base_eth
    critical.sort(key=lambda x: x['eth_score'], reverse=True)

    results = []
    allocated_indices = set()

    # Phase 1: Critical patients to ICU
    for item in critical:
        patient = item['patient']
        disease = patient['disease']
        severity = item['severity']
        surv_prob = item['survival']
        assigned = False

        # 1. Main hospital ICU
        main_hosp = resources[resources['hospital_id'] == main_hospital_id].iloc[0]
        if can_allocate_icu(main_hosp, patient, emergency):
            dist = haversine(patient['lat'], patient['lon'], main_hosp['lat'], main_hosp['lon'])
            adjusted_surv = max(surv_prob - dist * 0.005, 0)
            hosp_cap = hospital_capacity_score(main_hosp, emergency)
            eth = (0.4 * severity + 0.3 * adjusted_surv + 0.2 * hosp_cap + 0.1 * 0.5)
            if disease == 'COVID-19':
                eth += 0.15
            elif disease == 'Cardiovascular Disease':
                eth += 0.1
            reason = f"Critical patient (cluster-based) with {disease}. Allocated to ICU at main hospital."
            results.append({
                'patient_id': patient['patient_id'],
                'disease': disease,
                'age': patient['age'],
                'severity_score': round(severity, 3),
                'survival_probability': round(adjusted_surv, 3),
                'assigned_hospital': main_hosp['hospital_name'],
                'distance_km': round(dist, 2),
                'allocation_status': 'ALLOCATE',
                'ethical_score': round(eth, 3),
                'reason': reason
            })
            idx_h = resources[resources['hospital_id'] == main_hospital_id].index[0]
            resources.loc[idx_h, 'available_icu_beds'] -= 1
            resources.loc[idx_h, 'oxygen_supply'] -= 1.0
            if needs_ventilator(patient):
                resources.loc[idx_h, 'available_ventilators'] -= 1
            allocated_indices.add(item['idx'])
            assigned = True
            continue

        if not assigned:
            # 2. Other hospitals' ICU
            best_hosp = None
            best_dist = None
            best_adj_surv = None
            best_eth = -1
            other_hospitals = resources[resources['hospital_id'] != main_hospital_id]
            for _, hosp in other_hospitals.iterrows():
                if not can_allocate_icu(hosp, patient, emergency):
                    continue
                if disease != 'COVID-19':
                    spec_map = {'Cardiovascular Disease':'Cardiology','Nephropathy':'Nephrology',
                                'Neurological Disorder':'Neurology','Musculoskeletal Disorder':'Orthopedics',
                                'Hepatitis':'Gastroenterology'}
                    required = spec_map.get(disease)
                    if required:
                        specializations = [s.strip() for s in hosp['specializations'].split(',')]
                        if required not in specializations:
                            continue
                dist = haversine(patient['lat'], patient['lon'], hosp['lat'], hosp['lon'])
                adj_surv = max(surv_prob - dist * 0.005, 0)
                hosp_cap = hospital_capacity_score(hosp, emergency)
                eth = (0.4 * severity + 0.3 * adj_surv + 0.2 * hosp_cap + 0.1 * 0.5)
                if disease == 'COVID-19':
                    eth += 0.15
                elif disease == 'Cardiovascular Disease':
                    eth += 0.1
                if eth > best_eth:
                    best_eth = eth
                    best_hosp = hosp
                    best_dist = dist
                    best_adj_surv = adj_surv
            if best_hosp is not None:
                reason = f"Critical patient (cluster-based) with {disease}. Reallocated to {best_hosp['hospital_name']}."
                results.append({
                    'patient_id': patient['patient_id'],
                    'disease': disease,
                    'age': patient['age'],
                    'severity_score': round(severity, 3),
                    'survival_probability': round(best_adj_surv, 3),
                    'assigned_hospital': best_hosp['hospital_name'],
                    'distance_km': round(best_dist, 2),
                    'allocation_status': 'REALLOCATE',
                    'ethical_score': round(best_eth, 3),
                    'reason': reason
                })
                idx_h = resources[resources['hospital_id'] == best_hosp['hospital_id']].index[0]
                resources.loc[idx_h, 'available_icu_beds'] -= 1
                resources.loc[idx_h, 'oxygen_supply'] -= 1.0
                if needs_ventilator(patient):
                    resources.loc[idx_h, 'available_ventilators'] -= 1
                allocated_indices.add(item['idx'])
                assigned = True

        if not assigned:
            # 3. Fallback to normal ward (main hospital only)
            main_hosp = resources[resources['hospital_id'] == main_hospital_id].iloc[0]
            if main_hosp['available_beds'] > 0:
                dist = haversine(patient['lat'], patient['lon'], main_hosp['lat'], main_hosp['lon'])
                adj_surv = max(surv_prob - dist * 0.005, 0)
                eth = 0.4 * severity + 0.3 * adj_surv + 0.2 * 0.5 + 0.1 * 0.5
                reason = f"Critical patient but no ICU bed anywhere. Admitted to normal ward temporarily."
                results.append({
                    'patient_id': patient['patient_id'],
                    'disease': disease,
                    'age': patient['age'],
                    'severity_score': round(severity, 3),
                    'survival_probability': round(adj_surv, 3),
                    'assigned_hospital': main_hosp['hospital_name'],
                    'distance_km': round(dist, 2),
                    'allocation_status': 'WARD_ALLOCATED',
                    'ethical_score': round(eth, 3),
                    'reason': reason
                })
                idx_h = resources[resources['hospital_id'] == main_hospital_id].index[0]
                resources.loc[idx_h, 'available_beds'] -= 1
                oxygen_used = oxygen_consumption_for_ward(patient)
                if oxygen_used > 0:
                    resources.loc[idx_h, 'oxygen_supply'] -= oxygen_used
                allocated_indices.add(item['idx'])
            else:
                reason = "Critical patient but no ICU or ward beds → Emergency overflow."
                results.append({
                    'patient_id': patient['patient_id'],
                    'disease': disease,
                    'age': patient['age'],
                    'severity_score': round(severity, 3),
                    'survival_probability': 0.2,
                    'assigned_hospital': main_hosp['hospital_name'],
                    'distance_km': 0.0,
                    'allocation_status': 'EMERGENCY_OVERFLOW',
                    'ethical_score': 0.0,
                    'reason': reason
                })
                allocated_indices.add(item['idx'])

    # Phase 2: Mild/Moderate patients to normal ward (main hospital only)
    main_hosp = resources[resources['hospital_id'] == main_hospital_id].iloc[0]
    for item in non_critical:
        patient = item['patient']
        disease = patient['disease']
        severity = item['severity']
        surv_prob = item['survival']
        if main_hosp['available_beds'] > 0:
            dist = haversine(patient['lat'], patient['lon'], main_hosp['lat'], main_hosp['lon'])
            adj_surv = max(surv_prob - dist * 0.005, 0)
            eth = 0.4 * severity + 0.3 * adj_surv + 0.2 * 0.5 + 0.1 * 0.5
            reason = f"{item['group']} patient with {disease}. Allocated to normal ward at main hospital."
            results.append({
                'patient_id': patient['patient_id'],
                'disease': disease,
                'age': patient['age'],
                'severity_score': round(severity, 3),
                'survival_probability': round(adj_surv, 3),
                'assigned_hospital': main_hosp['hospital_name'],
                'distance_km': round(dist, 2),
                'allocation_status': 'WARD_ALLOCATED',
                'ethical_score': round(eth, 3),
                'reason': reason
            })
            idx_h = resources[resources['hospital_id'] == main_hospital_id].index[0]
            resources.loc[idx_h, 'available_beds'] -= 1
            oxygen_used = oxygen_consumption_for_ward(patient)
            if oxygen_used > 0:
                resources.loc[idx_h, 'oxygen_supply'] -= oxygen_used
        else:
            reason = f"{item['group']} patient but no ward beds → Emergency overflow."
            results.append({
                'patient_id': patient['patient_id'],
                'disease': disease,
                'age': patient['age'],
                'severity_score': round(severity, 3),
                'survival_probability': 0.2,
                'assigned_hospital': main_hosp['hospital_name'],
                'distance_km': 0.0,
                'allocation_status': 'EMERGENCY_OVERFLOW',
                'ethical_score': 0.0,
                'reason': reason
            })

    return pd.DataFrame(results), resources


# -------------------------------------------------------------------
# Standalone Evaluation (when running logic1.py directly)
# -------------------------------------------------------------------
def evaluate_and_print_metrics():
    """Train model on synthetic data and print all metrics to terminal."""
    print("\n" + "="*60)
    print("ICU ALLOCATION BACKEND - LOGISTIC REGRESSION EVALUATION")
    print("="*60)
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    df = generate_synthetic_training_data(n_samples=20000)
    X = df.drop('mortality_label', axis=1)
    y = df['mortality_label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Print results
    print("\n" + "-"*40)
    print("MODEL PERFORMANCE METRICS (Test Set)")
    print("-"*40)
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"ROC AUC:            {roc:.4f}")
    print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
    
    print("\n" + "-"*40)
    print("CLASSIFICATION REPORT")
    print("-"*40)
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Mortality']))
    
    print("="*60 + "\n")
    
    # Save model for later use (optional)
    if not os.path.exists(MODEL_PATH):
        print(f"Saving model to {MODEL_PATH}...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'feature_names': list(X.columns)
            }, f)
        print("Model saved.")
    else:
        print(f"Model already exists at {MODEL_PATH}. Not overwriting.")


if __name__ == "__main__":
    # When executed as a standalone script, run evaluation and print metrics.
    evaluate_and_print_metrics()