import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer

# Suppress warnings to guarantee clean IPC
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import path_resolver

# 1. Deterministic Categorical Dictionaries
PROTOCOL_TYPE_MAP = {
    'icmp': 0,
    'tcp': 1,
    'udp': 2
}

SERVICE_MAP = {
    'IRC': 0, 'X11': 1, 'Z39_50': 2, 'http_8001': 3, 'auth': 4,
    'bgp': 5, 'courier': 6, 'csnet_ns': 7, 'ctf': 8, 'daytime': 9,
    'discard': 10, 'domain': 11, 'domain_u': 12, 'echo': 13, 'eco_i': 14,
    'ecr_i': 15, 'efs': 16, 'exec': 17, 'finger': 18, 'ftp': 19,
    'ftp_data': 20, 'gopher': 21, 'harvest': 22, 'hostnames': 23, 'http': 24,
    'http_2784': 25, 'http_443': 26, 'aol': 27, 'imap4': 28, 'iso_tsap': 29,
    'klogin': 30, 'kshell': 31, 'ldap': 32, 'link': 33, 'login': 34,
    'mtp': 35, 'name': 36, 'netbios_dgm': 37, 'netbios_ns': 38, 'netbios_ssn': 39,
    'netstat': 40, 'nnsp': 41, 'nntp': 42, 'ntp_u': 43, 'other': 44,
    'pm_dump': 45, 'pop_2': 46, 'pop_3': 47, 'printer': 48, 'private': 49,
    'red_i': 50, 'remote_job': 51, 'rje': 52, 'shell': 53, 'smtp': 54,
    'sql_net': 55, 'ssh': 56, 'sunrpc': 57, 'supdup': 58, 'systat': 59,
    'telnet': 60, 'tftp_u': 61, 'tim_i': 62, 'time': 63, 'urh_i': 64,
    'urp_i': 65, 'uucp': 66, 'uucp_path': 67, 'vmnet': 68, 'whois': 69
}

FLAG_MAP = {
    'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4,
    'S0': 5, 'S1': 6, 'S2': 7, 'S3': 8, 'SF': 9, 'SH': 10
}

# 2. Attack Descriptions
ATTACK_DESCRIPTIONS = {
    'dos': 'A Denial-of-Service (DoS) attack is an attack meant to shut down a machine or network, making it inaccessible to its intended users. DoS attacks accomplish this by flooding the target with traffic, or sending it information that triggers a crash.',
    'probe': 'Probing is an attack where the intruder scans network devices to determine weaknesses in topology design or open ports to use for unauthorized access.',
    'r2l': 'Remote-to-Local (R2L) is an attack where an unauthorized remote attacker sends packets to a system to gain local access as a user.',
    'u2r': 'User-to-Root (U2R) is an attack where an attacker with regular user account access exploits vulnerabilities to gain root/administrative privileges.',
    'normal': 'This traffic pattern is verified as benign and normal.'
}

# 3. Standard Feature Sets
SELECTED_FEATURES_16 = [
    'protocol_type', 'service', 'flag', 'logged_in', 'count',
    'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_serror_rate', 'dst_host_rerror_rate'
]

NSL_KDD_COLUMNS_42 = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]

_GLOBAL_SCALER = None

def get_fitted_scaler():
    """Initializes or returns a globally fitted MinMaxScaler using validation data."""
    global _GLOBAL_SCALER
    if _GLOBAL_SCALER is not None:
        return _GLOBAL_SCALER

    scaler_path = path_resolver.resolve_model('scaler.sav')
    if scaler_path.exists():
        try:
            import pickle
            _GLOBAL_SCALER = pickle.load(open(scaler_path, 'rb'))
            return _GLOBAL_SCALER
        except Exception:
            pass

    val_csv_path = path_resolver.resolve_path('fs_new validation project.csv')
    
    if val_csv_path.exists():
        df = pd.read_csv(str(val_csv_path), header=None)
        if df.shape[1] >= 17:
            df = df.iloc[:, 0:16]
        df.columns = SELECTED_FEATURES_16
        
        # Deterministically encode categorical columns
        df['protocol_type'] = df['protocol_type'].map(lambda x: PROTOCOL_TYPE_MAP.get(str(x).lower().strip(), 1))
        df['service'] = df['service'].map(lambda x: SERVICE_MAP.get(str(x).strip(), 44))
        df['flag'] = df['flag'].map(lambda x: FLAG_MAP.get(str(x).strip(), 9))
        
        scaler = MinMaxScaler()
        scaler.fit(df.astype(float))
        _GLOBAL_SCALER = scaler
        return _GLOBAL_SCALER
    else:
        scaler = MinMaxScaler()
        dummy = np.zeros((2, 16))
        dummy[1, :] = 1.0
        scaler.fit(dummy)
        _GLOBAL_SCALER = scaler
        return _GLOBAL_SCALER

def encode_feature_vector(row_values):
    """
    Encodes 16 raw features into a numeric vector.
    Accepts list/tuple of 16 elements.
    """
    encoded = []
    p_val = str(row_values[0]).lower().strip()
    encoded.append(PROTOCOL_TYPE_MAP.get(p_val, 1))
    
    s_val = str(row_values[1]).strip()
    encoded.append(SERVICE_MAP.get(s_val, 44))
    
    f_val = str(row_values[2]).strip()
    encoded.append(FLAG_MAP.get(f_val, 9))
    
    for val in row_values[3:16]:
        try:
            encoded.append(float(val))
        except (ValueError, TypeError):
            encoded.append(0.0)
            
    return np.array(encoded, dtype=float)

def extract_features_from_csv(csv_path):
    """
    Intelligently reads an uploaded CSV file, supporting:
    - 42-column full NSL-KDD dataset (with or without headers)
    - 41-column NSL-KDD test dataset (with or without headers)
    - 16-column pre-selected feature dataset (with or without headers)
    Returns: (processed_dataframe_with_16_features, original_dataframe)
    """
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        
    has_header = any(col in first_line.lower() for col in ['protocol_type', 'duration', 'service', 'flag'])
    header_param = 0 if has_header else None
    
    df_raw = pd.read_csv(csv_path, header=header_param)
    
    if has_header:
        df_cols_lower = [str(c).lower().strip() for c in df_raw.columns]
        if all(feat in df_cols_lower for feat in SELECTED_FEATURES_16):
            col_map = {col: str(col).lower().strip() for col in df_raw.columns}
            df_renamed = df_raw.rename(columns=col_map)
            df_features = df_renamed[SELECTED_FEATURES_16].copy()
        elif 'duration' in df_cols_lower:
            col_map = {col: str(col).lower().strip() for col in df_raw.columns}
            df_renamed = df_raw.rename(columns=col_map)
            available = [f for f in SELECTED_FEATURES_16 if f in df_renamed.columns]
            df_features = df_renamed[available].copy()
        else:
            if df_raw.shape[1] >= 41:
                df_raw.columns = NSL_KDD_COLUMNS_42[:df_raw.shape[1]]
                df_features = df_raw[SELECTED_FEATURES_16].copy()
            elif df_raw.shape[1] >= 16:
                df_features = df_raw.iloc[:, 0:16].copy()
                df_features.columns = SELECTED_FEATURES_16
            else:
                raise ValueError(f"Insufficient columns: found {df_raw.shape[1]}, require at least 16 features.")
    else:
        if df_raw.shape[1] >= 41:
            df_raw.columns = NSL_KDD_COLUMNS_42[:df_raw.shape[1]]
            df_features = df_raw[SELECTED_FEATURES_16].copy()
        elif df_raw.shape[1] >= 16:
            df_features = df_raw.iloc[:, 0:16].copy()
            df_features.columns = SELECTED_FEATURES_16
        else:
            raise ValueError(f"Insufficient columns: found {df_raw.shape[1]}, require at least 16 features.")
            
    # Deterministically encode categorical features
    df_features['protocol_type'] = df_features['protocol_type'].map(lambda x: PROTOCOL_TYPE_MAP.get(str(x).lower().strip(), 1))
    df_features['service'] = df_features['service'].map(lambda x: SERVICE_MAP.get(str(x).strip(), 44))
    df_features['flag'] = df_features['flag'].map(lambda x: FLAG_MAP.get(str(x).strip(), 9))
    
    # Ensure all numeric columns are float
    for col in SELECTED_FEATURES_16[3:]:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0.0)
        
    return df_features, df_raw


# MITRE ATT&CK Enterprise Matrix & SOC Playbooks
MITRE_ATTACK_FRAMEWORK = {
    'dos': {
        'category': 'DoS (Denial of Service)',
        'tactic': 'Impact (TA0040)',
        'technique_id': 'T1498',
        'technique_name': 'Network Denial of Service',
        'sub_techniques': 'T1498.001 (Direct Network Flood), T1499 (Endpoint DoS)',
        'severity': 'CRITICAL',
        'cvss_score': '8.6',
        'impact': 'Exhaustion of network socket pools, memory exhaustion, packet loss, and service unresponsiveness.',
        'containment_playbook': [
            'sudo iptables -I INPUT -s <ATTACKER_IP> -j DROP',
            'sudo sysctl -w net.ipv4.tcp_syncookies=1',
            'sudo sysctl -w net.ipv4.tcp_max_syn_backlog=4096',
            'New-NetFirewallRule -DisplayName "NIDS-Block-DoS" -Direction Inbound -Action Block'
        ]
    },
    'probe': {
        'category': 'Probe (Reconnaissance)',
        'tactic': 'Discovery (TA0007)',
        'technique_id': 'T1046',
        'technique_name': 'Network Service Discovery',
        'sub_techniques': 'T1595 (Active Scanning), T1046 (Port Sweep)',
        'severity': 'HIGH',
        'cvss_score': '6.5',
        'impact': 'Systematic enumeration of listening TCP/UDP daemon ports and OS banner scraping.',
        'containment_playbook': [
            'sudo iptables -A INPUT -p tcp -s <ATTACKER_IP> -j TARPIT',
            'sudo iptables -A OUTPUT -p tcp --tcp-flags RST RST -m limit --limit 2/s -j ACCEPT',
            'sudo iptables -A OUTPUT -p tcp --tcp-flags RST RST -j DROP',
            'New-NetFirewallRule -DisplayName "NIDS-Block-Probe" -Direction Inbound -Action Block'
        ]
    },
    'r2l': {
        'category': 'R2L (Remote to Local)',
        'tactic': 'Initial Access (TA0001)',
        'technique_id': 'T1078',
        'technique_name': 'Valid Accounts / Credential Infiltration',
        'sub_techniques': 'T1110 (Brute Force), T1190 (Exploit Public-Facing Application)',
        'severity': 'HIGH',
        'cvss_score': '8.1',
        'impact': 'Unauthorized boundary infiltration from external untrusted IP into local user account.',
        'containment_playbook': [
            'sudo pkill -u <SUSPECT_USER> -9',
            'sudo passwd -l <SUSPECT_USER>',
            'sudo fail2ban-client set sshd banip <ATTACKER_IP>',
            'Revoke active OAuth tokens and rotate credentials immediately'
        ]
    },
    'u2r': {
        'category': 'U2R (User to Root)',
        'tactic': 'Privilege Escalation (TA0004)',
        'technique_id': 'T1068',
        'technique_name': 'Exploitation for Privilege Escalation',
        'sub_techniques': 'T1548 (Abuse Elevation Control Mechanism), T1068 (Kernel Exploit)',
        'severity': 'CRITICAL',
        'cvss_score': '9.8',
        'impact': 'Unprivileged local user gained UID 0 (root/SYSTEM), enabling rootkit and persistent kernel compromise.',
        'containment_playbook': [
            'sudo ip link set dev eth0 down',
            'sudo dd if=/dev/fmem of=/mnt/evidence/ram_dump.raw bs=1M status=progress',
            'sudo kill -9 <SUSPECT_PID>',
            'Quarantine host volume and redeploy from verified immutable baseline'
        ]
    },
    'normal': {
        'category': 'Normal (Benign Traffic)',
        'tactic': 'Routine Operations',
        'technique_id': 'BENIGN-001',
        'technique_name': 'Legitimate Network Telemetry',
        'sub_techniques': 'RFC-Compliant Protocol Transactions',
        'severity': 'SAFE',
        'cvss_score': '0.0',
        'impact': 'Standard verified network transactions adhering to baseline parameters.',
        'containment_playbook': [
            '# No containment required.',
            '# Traffic complies with organizational baseline policies.'
        ]
    }
}


def get_mitre_telemetry(attack_category: str) -> dict:
    """Return MITRE ATT&CK taxonomy object for a given attack classification."""
    cat_key = str(attack_category).strip().lower()
    return MITRE_ATTACK_FRAMEWORK.get(cat_key, MITRE_ATTACK_FRAMEWORK['normal'])

