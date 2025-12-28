import os
import sys

# Determine base directory
if os.path.exists(r'D:/network_ids'):
    BASE_DIR = r'D:/network_ids'
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'ids_model.pkl')

# Log paths
LOG_DIR = os.path.join(BASE_DIR, 'logs')
ALERT_LOG = os.path.join(LOG_DIR, 'alerts.log')
SYSTEM_LOG = os.path.join(LOG_DIR, 'system.log')

DATASET_URL = "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download"
TRAIN_FILE = os.path.join(RAW_DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_FILE = os.path.join(RAW_DATA_DIR, 'UNSW_NB15_testing-set.csv')

# Feature columns (49 features in UNSW-NB15)
FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

# Target columns
LABEL_COLUMN = 'label'  # Binary: 0 (normal) or 1 (attack)
ATTACK_CAT_COLUMN = 'attack_cat'  # Attack category

# Attack types in UNSW-NB15
ATTACK_TYPES = [
    'Normal',
    'Fuzzers',      # Attempts to discover security loopholes
    'Analysis',     # Port scanning, spam, html file penetration
    'Backdoors',    # Bypassing normal authentication
    'DoS',          # Denial of Service
    'Exploits',     # Exploiting security vulnerabilities
    'Generic',      # Attacks against block ciphers
    'Reconnaissance', # Information gathering
    'Shellcode',    # Exploiting software vulnerabilities
    'Worms'         # Self-replicating malware
]

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Alert Thresholds
ANOMALY_THRESHOLD = 0.5  # Threshold for anomaly score
ALERT_CONFIDENCE_THRESHOLD = 0.7

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
