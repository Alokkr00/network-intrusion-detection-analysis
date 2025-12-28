import argparse
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.data_loader import load_data
from utils.feature_engineering import FeatureEngineer
from detection.model_trainer import IDSModelTrainer
from sklearn.model_selection import train_test_split
import config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Random Forest with 10k samples
  python train_model.py --model random_forest --sample 10000
  
  # Train Neural Network with full dataset
  python train_model.py --model mlp --sample 0
  
  # Train Isolation Forest for anomaly detection
  python train_model.py --model isolation_forest --sample 50000
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'mlp', 'isolation_forest', 'one_class_svm'],
        help='Model type to train'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=10000,
        help='Number of samples to use (0 for all data)'
    )
    
    parser.add_argument(
        '--features',
        type=int,
        default=30,
        help='Number of features to select'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Disable feature selection'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model'
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("  NETWORK IDS - MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Sample size: {'All data' if args.sample == 0 else f'{args.sample:,}'}")
    print(f"  Features: {args.features}")
    print(f"  Feature selection: {'Disabled' if args.no_feature_selection else 'Enabled'}")
    print(f"  Test size: {args.test_size}")
    print("="*70 + "\n")
    
    # Load data
    print("Step 1: Loading data...")
    sample_size = None if args.sample == 0 else args.sample
    train_df = load_data(train=True, sample_size=sample_size)
    
    if train_df is None:
        print("\nFailed to load data. Please check dataset location.")
        return
    
    print(f"Loaded {len(train_df)} samples\n")
    
    # Preprocess
    print("Step 2: Preprocessing data...")
    fe = FeatureEngineer()
    X, y = fe.preprocess(
        train_df,
        fit=True,
        select_features=not args.no_feature_selection,
        k_features=args.features
    )
    print(f"Preprocessing complete! Final shape: {X.shape}\n")
    
    # Split data
    print("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples\n")
    
    # Train model
    print(f"Step 4: Training {args.model}...")
    trainer = IDSModelTrainer(model_type=args.model)
    history = trainer.train(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    print(f"Training time: {history['training_time']:.2f} seconds")
    
    print("\nTraining Metrics:")
    for metric, value in history['train_metrics'].items():
        if isinstance(value, (int, float)) and metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    if history['val_metrics']:
        print("\nTest Metrics:")
        for metric, value in history['val_metrics'].items():
            if isinstance(value, (int, float)) and metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
    
    print("="*70 + "\n")
    
    # Save model
    if args.save:
        print("Step 5: Saving model...")
        fe.save()
        trainer.save_model()
        print("Model and preprocessor saved!\n")
    else:
        print("Model not saved (use --save flag to save)\n")
    
    print("Training complete. \n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nFor help, run: python train_model.py --help")
        sys.exit(1)