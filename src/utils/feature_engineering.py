import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Import config from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

class FeatureEngineer:
    """
    Feature engineering pipeline for UNSW-NB15 dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
        
    def encode_categorical(self, df, fit=True):
        """
        Encode categorical features using Label Encoding
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with categorical features
        fit : bool
            Whether to fit the encoder (True for training, False for test)
        
        Returns:
        --------
        pd.DataFrame : Dataset with encoded features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols 
                          if col not in ['label', 'attack_cat']]
        
        print(f"Encoding {len(categorical_cols)} categorical features...")
        
        for col in categorical_cols:
            if fit:
                # Create and fit new encoder
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    # Handle unseen labels
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ 
                        else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with potential missing values
        
        Returns:
        --------
        pd.DataFrame : Dataset with imputed values
        """
        df = df.copy()
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Handling {missing.sum()} missing values...")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def remove_infinite_values(self, df):
        """
        Remove infinite values from numerical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with potential infinite values
        
        Returns:
        --------
        pd.DataFrame : Dataset with infinite values replaced
        """
        df = df.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Replace infinity with NaN
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            # Fill NaN with median
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        Scale numerical features using StandardScaler
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to scale
        fit : bool
            Whether to fit the scaler (True for training, False for test)
        
        Returns:
        --------
        pd.DataFrame : Scaled dataset
        """
        df = df.copy()
        
        # Get numerical columns (excluding label and attack_cat)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols 
                         if col not in ['label', 'attack_cat', 'id']]
        
        print(f"Scaling {len(numerical_cols)} numerical features...")
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def select_features(self, X, y, k=30, fit=True):
        """
        Select top k features using statistical tests
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        k : int
            Number of features to select
        fit : bool
            Whether to fit the selector
        
        Returns:
        --------
        pd.DataFrame : Selected features
        """
        if fit:
            print(f"Selecting top {k} features...")
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            feature_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[feature_mask].tolist()
            
            print(f"Selected features: {self.selected_features}")
            
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        else:
            if self.selected_features:
                return X[self.selected_features]
            return X
    
    def preprocess(self, df, fit=True, select_features=False, k_features=30):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset
        fit : bool
            Whether to fit transformers (True for training, False for test)
        select_features : bool
            Whether to perform feature selection
        k_features : int
            Number of features to select
        
        Returns:
        --------
        X : pd.DataFrame
            Preprocessed features
        y : pd.Series
            Target labels (if present)
        """
        df = df.copy()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Handle missing values
        print("\n1. Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Step 2: Remove infinite values
        print("2. Removing infinite values...")
        df = self.remove_infinite_values(df)
        
        # Step 3: Encode categorical features
        print("3. Encoding categorical features...")
        df = self.encode_categorical(df, fit=fit)
        
        # Step 4: Separate features and target
        if 'label' in df.columns:
            y = df['label']
            X = df.drop(['label'], axis=1)
            if 'attack_cat' in X.columns:
                X = X.drop(['attack_cat'], axis=1)
            if 'id' in X.columns:
                X = X.drop(['id'], axis=1)
        else:
            y = None
            X = df.copy()
            if 'attack_cat' in X.columns:
                X = X.drop(['attack_cat'], axis=1)
            if 'id' in X.columns:
                X = X.drop(['id'], axis=1)
        
        # Step 5: Scale features
        print("4. Scaling features...")
        X = self.scale_features(X, fit=fit)
        
        # Step 6: Feature selection (optional)
        if select_features and y is not None:
            print("5. Selecting features...")
            X = self.select_features(X, y, k=k_features, fit=fit)
        
        print("\n" + "=" * 60)
        print(f"Preprocessing complete!")
        print(f"Final shape: {X.shape}")
        print("=" * 60 + "\n")
        
        return X, y
    
    def save(self, path=None):
        """
        Save feature engineering pipeline
        
        Parameters:
        -----------
        path : str
            Path to save the pipeline
        """
        if path is None:
            path = config.SCALER_PATH
        
        pipeline = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }
        
        joblib.dump(pipeline, path)
        print(f"Feature engineering pipeline saved to {path}")
    
    def load(self, path=None):
        """
        Load feature engineering pipeline
        
        Parameters:
        -----------
        path : str
            Path to load the pipeline from
        """
        if path is None:
            path = config.SCALER_PATH
        
        pipeline = joblib.load(path)
        
        self.scaler = pipeline['scaler']
        self.label_encoders = pipeline['label_encoders']
        self.feature_selector = pipeline.get('feature_selector')
        self.selected_features = pipeline.get('selected_features')
        
        print(f"Feature engineering pipeline loaded from {path}")

if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering...")
    
    from utils.data_loader import load_data
    
    # Load sample data
    df = load_data(train=True, sample_size=1000)
    
    if df is not None:
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Preprocess data
        X, y = fe.preprocess(df, fit=True, select_features=True, k_features=20)
        
        print("\nPreprocessed Features:")
        print(X.head())
        print("\nTarget Distribution:")
        print(y.value_counts())
