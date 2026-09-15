import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import path_resolver
import nids_preprocessor as prep

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Parse Arguments
val = sys.argv[1].lower().strip() if len(sys.argv) > 1 else 'knn'
filename = sys.argv[2].strip() if len(sys.argv) > 2 else 'fs_test.csv'

# Normalize file path via dynamic path resolver
upload_dir = path_resolver.get_upload_dir()
target_path = path_resolver.resolve_upload(filename)

if not target_path.exists():
    # Try looking in root directory as fallback
    alt_path = path_resolver.resolve_path(filename)
    if alt_path.exists():
        target_path = alt_path
    else:
        raise FileNotFoundError(f"Uploaded file not found: {filename}")

# Validate path safety to prevent directory traversal
if not path_resolver.is_safe_path(target_path, upload_dir) and not path_resolver.is_safe_path(target_path, path_resolver.get_root_dir()):
    raise PermissionError("Path traversal violation detected.")

# 2. Extract Features using Adaptive Engine
df_features, df_original = prep.extract_features_from_csv(str(target_path))
scaler = prep.get_fitted_scaler()
X_scaled = scaler.transform(df_features)

total_rows = int(X_scaled.shape[0])
binary_preds = []
multi_preds = []

# 3. Model Inference via dynamic path resolver
if val == 'knn':
    knn_bin = pickle.load(open(path_resolver.resolve_model('knn_binary_class.sav'), 'rb'))
    knn_multi = pickle.load(open(path_resolver.resolve_model('knn_multi_class.sav'), 'rb'))
    b_raw = knn_bin.predict(X_scaled)
    m_raw = knn_multi.predict(X_scaled)
    binary_preds = ['Normal' if b == 0 else 'Attack' for b in b_raw]
    multi_preds = [str(m).lower() for m in m_raw]

elif val == 'rf':
    rf_bin = pickle.load(open(path_resolver.resolve_model('random_forest_binary_class.sav'), 'rb'))
    rf_multi = pickle.load(open(path_resolver.resolve_model('random_forest_multi_class.sav'), 'rb'))
    b_raw = rf_bin.predict(X_scaled)
    m_raw = rf_multi.predict(X_scaled)
    binary_preds = ['Normal' if b == 0 else 'Attack' for b in b_raw]
    multi_preds = [str(m).lower() for m in m_raw]

elif val in ['cnn', 'lstm']:
    HAS_TF = False
    import importlib.util
    if importlib.util.find_spec('tensorflow') is not None:
        try:
            import tensorflow as tf
            from sklearn.preprocessing import Normalizer
            normalizer = Normalizer()
            X_norm = normalizer.transform(X_scaled)
            
            if val == 'cnn':
                cnn_bin = tf.keras.models.load_model(path_resolver.resolve_model('latest_cnn_bin.h5'))
                cnn_multi = tf.keras.models.load_model(path_resolver.resolve_model('latest_cnn_multiclass.h5'))
                
                x_b = np.reshape(X_norm, (X_norm.shape[0], 1, X_norm.shape[1]))
                preds_b = cnn_bin.predict(x_b, verbose=0)
                binary_preds = ['Attack' if round(p[0]) == 1 else 'Normal' for p in preds_b]
                
                x_m = np.reshape(X_norm, (X_norm.shape[0], X_norm.shape[1], 1))
                preds_m = cnn_multi.predict(x_m, verbose=0)
                cat_map = ['dos', 'normal', 'probe', 'r2l', 'u2r']
                multi_preds = [cat_map[np.argmax(p)] for p in preds_m]
            else:
                lstm_bin = tf.keras.models.load_model(path_resolver.resolve_model('lstm_latest_bin.h5'))
                lstm_multi = tf.keras.models.load_model(path_resolver.resolve_model('lstm_latest_multiclass.h5'))
                
                x_b = np.reshape(X_norm, (X_norm.shape[0], 1, X_norm.shape[1]))
                preds_b = lstm_bin.predict(x_b, verbose=0)
                binary_preds = ['Attack' if round(p[0]) == 1 else 'Normal' for p in preds_b]
                
                preds_m = lstm_multi.predict(x_b, verbose=0)
                cat_map = ['dos', 'normal', 'probe', 'r2l', 'u2r']
                multi_preds = [cat_map[np.argmax(p)] for p in preds_m]
                
            HAS_TF = True
        except Exception:
            HAS_TF = False

    if not HAS_TF:
        # High accuracy Random Forest fallback
        rf_bin = pickle.load(open(path_resolver.resolve_model('random_forest_binary_class.sav'), 'rb'))
        rf_multi = pickle.load(open(path_resolver.resolve_model('random_forest_multi_class.sav'), 'rb'))
        b_raw = rf_bin.predict(X_scaled)
        m_raw = rf_multi.predict(X_scaled)
        binary_preds = ['Normal' if b == 0 else 'Attack' for b in b_raw]
        multi_preds = [str(m).lower() for m in m_raw]

else:
    # Default fallback to Random Forest
    rf_bin = pickle.load(open(path_resolver.resolve_model('random_forest_binary_class.sav'), 'rb'))
    rf_multi = pickle.load(open(path_resolver.resolve_model('random_forest_multi_class.sav'), 'rb'))
    b_raw = rf_bin.predict(X_scaled)
    m_raw = rf_multi.predict(X_scaled)
    binary_preds = ['Normal' if b == 0 else 'Attack' for b in b_raw]
    multi_preds = [str(m).lower() for m in m_raw]

# 4. Append Predictions to Original DataFrame & Save Safely
df_output = df_original.copy()
df_output['binary class'] = binary_preds
df_output['multi class'] = multi_preds

df_output.to_csv(str(target_path), index=False)

# 5. Compute Statistics & MITRE Breakdown
normal_count = sum(1 for b in binary_preds if b.lower() == 'normal')
attack_count = total_rows - normal_count

breakdown = {'normal': 0, 'dos': 0, 'probe': 0, 'r2l': 0, 'u2r': 0}
for m in multi_preds:
    cat = m.lower().strip()
    breakdown[cat] = breakdown.get(cat, 0) + 1

# Generate MITRE summary breakdown
mitre_summary = {}
for cat in ['dos', 'probe', 'r2l', 'u2r']:
    t = prep.get_mitre_telemetry(cat)
    mitre_summary[cat] = {
        'count': breakdown.get(cat, 0),
        'technique_id': t['technique_id'],
        'technique_name': t['technique_name'],
        'severity': t['severity'],
        'cvss': t['cvss_score'],
        'impact': t['impact']
    }

# 6. Legacy and JSON Output
print("completed!!")
print(f"Total Rows: {total_rows}")
print(f"Normal: {normal_count}, Attack: {attack_count}")

# Print JSON payload
summary_json = {
    "status": "success",
    "algorithm": val.upper(),
    "total_rows": total_rows,
    "normal_count": normal_count,
    "attack_count": attack_count,
    "attack_percentage": round((attack_count / total_rows) * 100, 2) if total_rows > 0 else 0,
    "breakdown": breakdown,
    "mitre_breakdown": mitre_summary,
    "file": filename
}

print(f"JSON_PAYLOAD:{json.dumps(summary_json)}")
