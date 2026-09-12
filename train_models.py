import os
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import nids_preprocessor as prep

base_dir = os.path.dirname(os.path.abspath(__file__))
backup_dir = os.path.join(base_dir, 'models_backup')
os.makedirs(backup_dir, exist_ok=True)

# 1. Backup old models if not already backed up
for fname in ['knn_binary_class.sav', 'knn_multi_class.sav', 'random_forest_binary_class.sav', 'random_forest_multi_class.sav']:
    src = os.path.join(base_dir, fname)
    dst = os.path.join(backup_dir, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Backed up {fname} to {backup_dir}")

# 2. Map sub-attacks to 4 high-level categories
ATTACK_MAPPING = {
    'normal': 'normal',
    # DoS
    'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos',
    'pod': 'dos', 'land': 'dos', 'apache2': 'dos', 'mailbomb': 'dos',
    'processtable': 'dos', 'udpstorm': 'dos',
    # Probe
    'satan': 'probe', 'ipsweep': 'probe', 'portsweep': 'probe',
    'nmap': 'probe', 'mscan': 'probe', 'saint': 'probe',
    # R2L
    'guess_passwd': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l',
    'snmpguess': 'r2l', 'snmpgetattack': 'r2l', 'httptunnel': 'r2l',
    'multihop': 'r2l', 'named': 'r2l', 'sendmail': 'r2l', 'ftp_write': 'r2l',
    'imap': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'worm': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l',
    # U2R
    'rootkit': 'u2r', 'xterm': 'u2r', 'buffer_overflow': 'u2r', 'ps': 'u2r',
    'loadmodule': 'u2r', 'perl': 'u2r', 'sqlattack': 'u2r'
}

# 3. Load validation dataset
val_path = os.path.join(base_dir, 'fs_new validation project.csv')
df = pd.read_csv(val_path, header=None)
X_raw = df.iloc[:, 0:16].copy()
X_raw.columns = prep.SELECTED_FEATURES_16
y_raw = df.iloc[:, 16].astype(str).str.strip().str.lower()

# Encode features
X_encoded = X_raw.copy()
X_encoded['protocol_type'] = X_encoded['protocol_type'].map(lambda x: prep.PROTOCOL_TYPE_MAP.get(str(x).lower().strip(), 1))
X_encoded['service'] = X_encoded['service'].map(lambda x: prep.SERVICE_MAP.get(str(x).strip(), 44))
X_encoded['flag'] = X_encoded['flag'].map(lambda x: prep.FLAG_MAP.get(str(x).strip(), 9))
for col in prep.SELECTED_FEATURES_16[3:]:
    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0.0)

# Scale features
scaler = prep.get_fitted_scaler()
X_scaled = scaler.transform(X_encoded)

# Labels
y_binary = np.array([0 if a == 'normal' else 1 for a in y_raw])
y_multi = np.array([ATTACK_MAPPING.get(a, 'probe') for a in y_raw])

print(f"Training on {X_scaled.shape[0]} samples...")
print(f"Binary distribution: Normal={np.sum(y_binary==0)}, Attack={np.sum(y_binary==1)}")
print(f"Multi distribution: {pd.Series(y_multi).value_counts().to_dict()}")

# 4. Fit Models
# KNN
knn_bin = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_bin.fit(X_scaled, y_binary)
pickle.dump(knn_bin, open(os.path.join(base_dir, 'knn_binary_class.sav'), 'wb'))
print("Saved knn_binary_class.sav")

knn_multi = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_multi.fit(X_scaled, y_multi)
pickle.dump(knn_multi, open(os.path.join(base_dir, 'knn_multi_class.sav'), 'wb'))
print("Saved knn_multi_class.sav")

# Random Forest
rf_bin = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_bin.fit(X_scaled, y_binary)
pickle.dump(rf_bin, open(os.path.join(base_dir, 'random_forest_binary_class.sav'), 'wb'))
print("Saved random_forest_binary_class.sav")

rf_multi = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_multi.fit(X_scaled, y_multi)
pickle.dump(rf_multi, open(os.path.join(base_dir, 'random_forest_multi_class.sav'), 'wb'))
print("Saved random_forest_multi_class.sav")

print("All models successfully trained and serialized with current scikit-learn!")
