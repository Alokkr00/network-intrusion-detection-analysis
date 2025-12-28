import pandas as pd
import numpy as np
import os
import sys
import urllib.request
from tqdm import tqdm

# Import config from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset():
    """
    Download UNSW-NB15 dataset
    Note: You may need to manually download from the official source
    """
    print("UNSW-NB15 Dataset Download Instructions:")
    print("=" * 60)
    print("Please manually download the dataset from:")
    print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("\nDownload these files:")
    print("1. UNSW_NB15_training-set.csv")
    print("2. UNSW_NB15_testing-set.csv")
    print(f"\nPlace them in: {config.RAW_DATA_DIR}")
    print("=" * 60)

def load_data(train=True, sample_size=None):
    """
    Load UNSW-NB15 dataset
    
    Parameters:
    -----------
    train : bool
        If True, load training set; otherwise load test set
    sample_size : int, optional
        Number of samples to load (for testing)
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    file_path = config.TRAIN_FILE if train else config.TEST_FILE
    
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}")
        download_dataset()
        return None
    
    print(f"Loading {'training' if train else 'testing'} data from {file_path}...")
    
    # Load data
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    return df

def get_data_info(df):
    """
    Display dataset information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    """
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\nShape: {df.shape}")
    print(f"Features: {df.shape[1]}")
    print(f"Samples: {df.shape[0]}")
    
    if 'label' in df.columns:
        print("\n--- Class Distribution ---")
        label_counts = df['label'].value_counts()
        print(f"Normal (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
        print(f"Attack (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
    
    if 'attack_cat' in df.columns:
        print("\n--- Attack Types Distribution ---")
        attack_dist = df['attack_cat'].value_counts()
        for attack, count in attack_dist.items():
            print(f"{attack}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\n--- Sample Data ---")
    print(df.head())
    
    print("=" * 60 + "\n")

def get_feature_stats(df):
    """
    Get statistical summary of features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    """
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical Features: {len(numerical_cols)}")
    print(df[numerical_cols].describe())
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical Features: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    
    # Load small sample for testing
    train_data = load_data(train=True, sample_size=1000)
    
    if train_data is not None:
        get_data_info(train_data)
        get_feature_stats(train_data)
