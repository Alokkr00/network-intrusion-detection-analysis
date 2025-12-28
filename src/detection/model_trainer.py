import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Import config from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

class IDSModelTrainer:
    """
    Train and evaluate machine learning models for intrusion detection
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        model_type : str
            Type of model to train:
            - 'random_forest': Random Forest (supervised)
            - 'gradient_boosting': Gradient Boosting (supervised)
            - 'mlp': Multi-Layer Perceptron Neural Network
            - 'isolation_forest': Isolation Forest (unsupervised anomaly detection)
            - 'one_class_svm': One-Class SVM (unsupervised anomaly detection)
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.training_history = {}
        
    def _initialize_model(self):
        """
        Initialize the selected model with optimal hyperparameters
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=config.RANDOM_STATE,
                verbose=1
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=config.RANDOM_STATE,
                verbose=True,
                early_stopping=True
            )
        
        elif self.model_type == 'isolation_forest':
            return IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Expected proportion of anomalies
                random_state=config.RANDOM_STATE,
                verbose=1,
                n_jobs=-1
            )
        
        elif self.model_type == 'one_class_svm':
            return OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.1  # Upper bound on fraction of outliers
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training labels
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features
        y_val : pd.Series or np.ndarray, optional
            Validation labels
        
        Returns:
        --------
        dict : Training metrics
        """
        print("\n" + "=" * 60)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Train model
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            # Unsupervised: train only on normal traffic
            normal_indices = y_train == 0
            X_normal = X_train[normal_indices]
            print(f"Training on {len(X_normal)} normal samples...")
            self.model.fit(X_normal)
        else:
            # Supervised: train on all data
            print(f"Training on {len(X_train)} samples...")
            self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train, dataset_name="Training")
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, dataset_name="Validation")
        
        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print("=" * 60 + "\n")
        
        return self.training_history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features to predict
        
        Returns:
        --------
        np.ndarray : Predictions (0 for normal, 1 for attack)
        """
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            # Anomaly detection models return -1 for anomalies, 1 for normal
            predictions = self.model.predict(X)
            # Convert to 0 (normal) and 1 (attack)
            predictions = np.where(predictions == 1, 0, 1)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features to predict
        
        Returns:
        --------
        np.ndarray : Prediction probabilities
        """
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            # For anomaly detection, use decision function as proxy
            scores = self.model.decision_function(X)
            # Normalize to [0, 1] range
            probabilities = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probabilities, probabilities])
        else:
            return self.model.predict_proba(X)
    
    def evaluate(self, X, y, dataset_name="Test"):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True labels
        dataset_name : str
            Name of the dataset being evaluated
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        print(f"\n--- {dataset_name} Set Evaluation ---")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # ROC-AUC (if probability predictions available)
        try:
            y_proba = self.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
        except:
            roc_auc = None
        
        # Print metrics
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Normal', 'Attack']))
        
        # Calculate attack detection rate per class
        tn, fp, fn, tp = cm.ravel()
        attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nAttack Detection Rate: {attack_detection_rate:.4f} ({tp}/{tp+fn})")
        print(f"False Alarm Rate: {false_alarm_rate:.4f} ({fp}/{fp+tn})")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'attack_detection_rate': attack_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, X, y, save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True labels
        save_path : str, optional
            Path to save the plot
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {self.model_type}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, X, y, save_path=None):
        """
        Plot ROC curve
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features
        y : pd.Series or np.ndarray
            True labels
        save_path : str, optional
            Path to save the plot
        """
        try:
            y_proba = self.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = roc_auc_score(y, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_type}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved to {save_path}")
            
            plt.tight_layout()
            return plt.gcf()
        except:
            print("Could not generate ROC curve")
            return None
    
    def get_feature_importance(self, feature_names=None, top_k=20):
        """
        Get feature importance (for tree-based models)
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_k : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame : Feature importance scores
        """
        if self.model_type not in ['random_forest', 'gradient_boosting']:
            print(f"Feature importance not available for {self.model_type}")
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_k)
        
        print(f"\nTop {top_k} Most Important Features:")
        print(importance_df)
        
        return importance_df
    
    def save_model(self, path=None):
        """
        Save trained model
        
        Parameters:
        -----------
        path : str, optional
            Path to save the model
        """
        if path is None:
            path = config.MODEL_PATH.replace('.pkl', f'_{self.model_type}.pkl')
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load trained model
        
        Parameters:
        -----------
        path : str, optional
            Path to load the model from
        """
        if path is None:
            path = config.MODEL_PATH.replace('.pkl', f'_{self.model_type}.pkl')
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    print("Model Trainer Module - Ready for use")
