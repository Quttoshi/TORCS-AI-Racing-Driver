#!/usr/bin/env python3
"""
TORCS AI Training Script - COMPLETELY FIXED VERSION
Resolves all variable naming conflicts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import r2_score as sklearn_r2_score  # FIXED: Rename to avoid conflicts
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import joblib
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def filter_quality_data(df, filename):
    """Filter out poor quality training data (like oval tracks)"""
    print(f"\n Analyzing data quality for: {filename}")
    
    if 'control_steer' not in df.columns:
        print(f"   REJECTED: No steering data found")
        return None
    
    steer_values = df['control_steer'].dropna()
    
    if len(steer_values) < 100:
        print(f"   REJECTED: Too few samples ({len(steer_values)})")
        return None
    
    # Calculate steering distribution
    left_turns = len(steer_values[steer_values < -0.1])
    straight = len(steer_values[abs(steer_values) <= 0.1])
    right_turns = len(steer_values[steer_values > 0.1])
    total = len(steer_values)
    
    # Calculate ratios
    straight_ratio = straight / total
    left_right_ratio = left_turns / max(right_turns, 1)
    turn_variety = (left_turns + right_turns) / total
    
    print(f"   Steering breakdown:")
    print(f"    Left: {left_turns} ({left_turns/total*100:.1f}%)")
    print(f"    Straight: {straight} ({straight_ratio*100:.1f}%)")
    print(f"    Right: {right_turns} ({right_turns/total*100:.1f}%)")
    print(f"    L/R ratio: {left_right_ratio:.3f}")
    print(f"    Turn variety: {turn_variety*100:.1f}%")
    
    # Quality thresholds
    MAX_STRAIGHT_RATIO = 0.85  # Max 85% straight driving
    MIN_LEFT_RIGHT_RATIO = 0.15  # At least 15% as many left turns as right
    MIN_TURN_VARIETY = 0.20  # At least 20% turning
    
    # Check quality criteria
    quality_issues = []
    
    if straight_ratio > MAX_STRAIGHT_RATIO:
        quality_issues.append(f"too much straight driving ({straight_ratio*100:.1f}%)")
    
    if left_right_ratio < MIN_LEFT_RIGHT_RATIO:
        quality_issues.append(f"severe left/right imbalance (ratio: {left_right_ratio:.3f})")
    
    if turn_variety < MIN_TURN_VARIETY:
        quality_issues.append(f"insufficient turn variety ({turn_variety*100:.1f}%)")
    
    # Speed analysis (if available)
    if 'speed_x' in df.columns:
        speeds = df['speed_x'].dropna()
        avg_speed = speeds.mean()
        print(f"    Average speed: {avg_speed:.1f} km/h")
        
        if avg_speed < 5:
            quality_issues.append(f"very low speeds ({avg_speed:.1f} km/h - mostly stationary)")
    
    # Distance analysis (if available)
    if 'distance_raced' in df.columns:
        distances = df['distance_raced'].dropna()
        if len(distances) > 0:
            total_distance = distances.max() - distances.min()
            print(f"    Distance covered: {total_distance/1000:.1f} km")
            
            if total_distance < 1000:
                quality_issues.append(f"limited distance coverage ({total_distance:.0f}m)")
    
    # Decision
    if quality_issues:
        print(f"   REJECTED: {', '.join(quality_issues)}")
        print(f"   This appears to be oval track or poor quality data")
        return None
    else:
        print(f"   ACCEPTED: Good quality data for training")
        return df

def load_and_combine_data(data_directory='telemetry_data', file_pattern="*.csv"):
    """Load and combine all CSV files with human driving data, filtering out poor quality data"""
    print(" TORCS AI Training - Data Loading with Quality Filter")
    print("=" * 60)
    print(" Looking for training data...")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_directory, file_pattern))
    
    if not csv_files:
        print(f" ERROR: No CSV files found in {data_directory}")
        return None
    
    print(f" Found {len(csv_files)} CSV files to analyze:")
    for file in csv_files:
        print(f"   - {os.path.basename(file)}")
    
    # Load and filter data
    accepted_dataframes = []
    rejected_files = []
    total_samples = 0
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            print(f"\n Loading: {filename} ({len(df)} samples)")
            
            # Apply quality filter
            filtered_df = filter_quality_data(df, filename)
            
            if filtered_df is not None:
                # Reset index to avoid conflicts
                filtered_df = filtered_df.reset_index(drop=True)
                accepted_dataframes.append(filtered_df)
                total_samples += len(filtered_df)
                print(f"    Added {len(filtered_df)} samples to training set")
            else:
                rejected_files.append(filename)
                
        except Exception as e:
            print(f"    ERROR loading {filename}: {e}")
            rejected_files.append(filename)
    
    # Summary
    print(f"\n DATA LOADING SUMMARY:")
    print(f"    Accepted: {len(accepted_dataframes)} files")
    print(f"    Rejected: {len(rejected_files)} files")
    
    if rejected_files:
        print(f"     Rejected files: {', '.join(rejected_files)}")
    
    if not accepted_dataframes:
        print(" ERROR: No quality data could be loaded!")
        return None
    
    # Find common columns across all accepted files
    common_columns = set(accepted_dataframes[0].columns)
    for df in accepted_dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    print(f"    Common columns: {len(common_columns)}")
    
    # Filter each dataframe to only include common columns
    filtered_dataframes = []
    for df in accepted_dataframes:
        filtered_df = df[list(common_columns)].copy()
        filtered_dataframes.append(filtered_df)
    
    # Combine all data
    try:
        combined_data = pd.concat(filtered_dataframes, ignore_index=True, sort=False)
        print(f"    FINAL DATASET: {len(combined_data)} samples, {len(combined_data.columns)} features")
        
        # Final quality check on combined data
        print(f"\n COMBINED DATA QUALITY CHECK:")
        steer_values = combined_data['control_steer'].dropna()
        left_turns = len(steer_values[steer_values < -0.1])
        straight = len(steer_values[abs(steer_values) <= 0.1])
        right_turns = len(steer_values[steer_values > 0.1])
        total = len(steer_values)
        
        print(f"   Left turns: {left_turns} ({left_turns/total*100:.1f}%)")
        print(f"   Straight: {straight} ({straight/total*100:.1f}%)")
        print(f"   Right turns: {right_turns} ({right_turns/total*100:.1f}%)")
        print(f"   L/R ratio: {left_turns/max(right_turns,1):.3f}")
        
        return combined_data
        
    except Exception as e:
        print(f" ERROR combining data: {e}")
        return None

def preprocess_data(data):
    """Clean and preprocess the data"""
    print("\n DATA PREPROCESSING")
    print("=" * 40)
    
    original_size = len(data)
    print(f" Original dataset size: {original_size}")
    
    # Check for required control columns
    control_columns = ['control_steer', 'control_accel', 'control_brake', 'control_gear']
    missing_controls = [col for col in control_columns if col not in data.columns]
    
    if missing_controls:
        print(f" ERROR: Missing required control columns: {missing_controls}")
        return None
    
    print(" All control columns present")
    
    # Remove rows with missing critical values
    data = data.dropna(subset=control_columns)
    print(f" After removing rows with missing critical data: {len(data)}")
    
    # Remove stationary data (car not moving)
    if 'speed_x' in data.columns:
        moving_mask = np.abs(data['speed_x']) > 0.5
        data = data[moving_mask]
        print(f" After removing stationary data: {len(data)}")
    
    # Remove extreme outliers
    outlier_columns = ['speed_x', 'track_position', 'angle', 'rpm']
    for col in outlier_columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.005)
            Q99 = data[col].quantile(0.995)
            data = data[(data[col] >= Q1) & (data[col] <= Q99)]
    
    print(f"  After removing outliers: {len(data)}")
    
    # Clean control data
    print(" Cleaning control data...")
    
    # Clean and clip control values
    data['control_steer'] = np.clip(data['control_steer'], -1.0, 1.0)
    data['control_accel'] = np.clip(data['control_accel'], 0.0, 1.0)
    data['control_brake'] = np.clip(data['control_brake'], 0.0, 1.0)
    
    # Clean gear data
    data['control_gear'] = pd.to_numeric(data['control_gear'], errors='coerce')
    data = data.dropna(subset=['control_gear'])
    data['control_gear'] = data['control_gear'].astype(int)
    
    valid_gear_mask = (data['control_gear'] >= 1) & (data['control_gear'] <= 6)
    invalid_gears = (~valid_gear_mask).sum()
    data = data[valid_gear_mask]
    if invalid_gears > 0:
        print(f"    Removed {invalid_gears} samples with invalid gears")
    
    # Show statistics
    print(f"\n CONTROL DATA STATISTICS:")
    for col in control_columns:
        if col == 'control_gear':
            gear_dist = data[col].value_counts().sort_index()
            print(f"    {col}: {dict(gear_dist)}")
        else:
            print(f"    {col}: mean={data[col].mean():.3f}, std={data[col].std():.3f}, range=[{data[col].min():.3f}, {data[col].max():.3f}]")
    
    # Final quality check
    final_steer = data['control_steer']
    final_left = len(final_steer[final_steer < -0.1])
    final_right = len(final_steer[final_steer > 0.1])
    final_ratio = final_left / max(final_right, 1)
    
    print(f"\n FINAL PROCESSED DATA QUALITY:")
    print(f"   Samples: {len(data)}")
    print(f"   Left/Right ratio: {final_ratio:.3f}")
    
    if len(data) < 1000:
        print(f"  WARNING: Only {len(data)} samples remaining")
    elif final_ratio < 0.3:
        print(f"  WARNING: Still some left/right imbalance")
    else:
        print(f" Good data quality for training!")
    
    return data

def prepare_features_and_targets(data):
    """Prepare feature matrix and target variables for training"""
    print("\n FEATURE AND TARGET PREPARATION")
    print("=" * 40)
    
    # Define feature columns
    feature_columns = []
    
    # Basic state features
    basic_features = ['speed_x', 'speed_y', 'speed_z', 'angle', 'track_position', 'rpm']
    for feature in basic_features:
        if feature in data.columns:
            feature_columns.append(feature)
    
    # Additional features
    additional_features = ['fuel', 'damage', 'distance_from_start']
    for feature in additional_features:
        if feature in data.columns:
            feature_columns.append(feature)
    
    # Track sensor features
    track_sensors = [col for col in data.columns if 'track_sensor_' in col]
    track_sensors = sorted(track_sensors)
    feature_columns.extend(track_sensors)
    
    # Wheel speed features
    wheel_features = [col for col in data.columns if 'wheel_speed_' in col]
    feature_columns.extend(wheel_features)
    
    # Current gear as a feature
    if 'gear' in data.columns:
        feature_columns.append('gear')
    
    print(f" Selected {len(feature_columns)} features:")
    basic_count = len([f for f in feature_columns if not f.startswith('track_sensor_')])
    print(f"    Basic features: {basic_count}")
    print(f"    Track sensors: {len(track_sensors)}")
    
    if track_sensors:
        sensor_angles = []
        for sensor in track_sensors:
            try:
                angle = int(sensor.split('_')[-1])
                sensor_angles.append(angle)
            except:
                pass
        if sensor_angles:
            print(f"    Sensor angles: {min(sensor_angles)}° to {max(sensor_angles)}°")
    
    # Create feature matrix
    X = data[feature_columns].copy()
    X = X.fillna(X.mean())  # Fill missing values
    
    # Define targets
    target_columns = ['control_steer', 'control_accel', 'control_brake', 'control_gear']
    y = data[target_columns].copy()
    
    print(f" Feature matrix shape: {X.shape}")
    print(f" Target matrix shape: {y.shape}")
    
    # Split targets
    continuous_targets = ['control_steer', 'control_accel', 'control_brake']
    discrete_targets = ['control_gear']
    
    y_continuous = y[continuous_targets]
    y_discrete = y[discrete_targets]
    
    return X, y_continuous, y_discrete, feature_columns, continuous_targets, discrete_targets

def train_driving_models(X_train, X_val, y_continuous_train, y_continuous_val, 
                        y_discrete_train, y_discrete_val):
    """Train models for all driving controls - FIXED VERSION"""
    print("\n TRAINING DRIVING MODELS")
    print("=" * 40)
    
    models = {}
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Method 1: Random Forest for continuous controls
    print(" Training Random Forest for continuous controls (steer, accel, brake)...")
    rf_continuous = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_continuous.fit(X_train, y_continuous_train)
    y_continuous_pred_rf = rf_continuous.predict(X_val)
    
    # FIXED: Use sklearn_r2_score consistently
    rf_continuous_mse = mean_squared_error(y_continuous_val, y_continuous_pred_rf)
    rf_continuous_r2 = sklearn_r2_score(y_continuous_val, y_continuous_pred_rf)
    
    print(f"    Random Forest Continuous - MSE: {rf_continuous_mse:.4f}, R²: {rf_continuous_r2:.4f}")
    
    # Method 2: Random Forest for gear
    print("  Training Random Forest for gear selection...")
    
    y_discrete_train_clean = y_discrete_train.copy().astype(int)
    y_discrete_val_clean = y_discrete_val.copy().astype(int)
    
    print(f"    Gear training data shape: {y_discrete_train_clean.shape}")
    print(f"    Unique gears in training: {np.unique(y_discrete_train_clean.values.ravel())}")
    
    rf_discrete = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_discrete.fit(X_train, y_discrete_train_clean.values.ravel())
    y_discrete_pred_rf = rf_discrete.predict(X_val)
    
    rf_gear_accuracy = accuracy_score(y_discrete_val_clean.values.ravel(), y_discrete_pred_rf)
    print(f"    Random Forest Gear - Accuracy: {rf_gear_accuracy:.4f}")
    
    models['random_forest'] = {
        'continuous': rf_continuous,
        'discrete': rf_discrete,
        'scaler': scaler
    }
    
    results['random_forest'] = {
        'continuous_mse': rf_continuous_mse,
        'continuous_r2': rf_continuous_r2,
        'gear_accuracy': rf_gear_accuracy
    }
    
    # Method 3: Neural Networks
    print(" Training Neural Networks...")
    
    # Neural network for continuous controls
    nn_continuous = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    nn_continuous.fit(X_train_scaled, y_continuous_train)
    y_continuous_pred_nn = nn_continuous.predict(X_val_scaled)
    
    # FIXED: Use sklearn_r2_score consistently
    nn_continuous_mse = mean_squared_error(y_continuous_val, y_continuous_pred_nn)
    nn_continuous_r2 = sklearn_r2_score(y_continuous_val, y_continuous_pred_nn)
    
    print(f"    Neural Network Continuous - MSE: {nn_continuous_mse:.4f}, R²: {nn_continuous_r2:.4f}")
    
    # Neural network for gear
    nn_discrete = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    nn_discrete.fit(X_train_scaled, y_discrete_train_clean.values.ravel())
    y_discrete_pred_nn = nn_discrete.predict(X_val_scaled)
    
    nn_gear_accuracy = accuracy_score(y_discrete_val_clean.values.ravel(), y_discrete_pred_nn)
    print(f"    Neural Network Gear - Accuracy: {nn_gear_accuracy:.4f}")
    
    models['neural_network'] = {
        'continuous': nn_continuous,
        'discrete': nn_discrete,
        'scaler': scaler
    }
    
    results['neural_network'] = {
        'continuous_mse': nn_continuous_mse,
        'continuous_r2': nn_continuous_r2,
        'gear_accuracy': nn_gear_accuracy
    }
    
    # Method 4: Unified approach
    print(" Training unified model...")
    
    y_all_train = np.column_stack([y_continuous_train.values, y_discrete_train_clean.values])
    y_all_val = np.column_stack([y_continuous_val.values, y_discrete_val_clean.values])
    
    rf_unified = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_unified.fit(X_train, y_all_train)
    y_all_pred = rf_unified.predict(X_val)
    
    # Round gear predictions
    y_all_pred_corrected = y_all_pred.copy()
    y_all_pred_corrected[:, 3] = np.round(np.clip(y_all_pred_corrected[:, 3], 1, 6))
    
    # FIXED: Use sklearn_r2_score consistently
    unified_mse = mean_squared_error(y_all_val, y_all_pred_corrected)
    unified_r2 = sklearn_r2_score(y_all_val, y_all_pred_corrected)
    unified_gear_accuracy = accuracy_score(y_all_val[:, 3], y_all_pred_corrected[:, 3])
    
    print(f"    Unified Model - MSE: {unified_mse:.4f}, R²: {unified_r2:.4f}, Gear Acc: {unified_gear_accuracy:.4f}")
    
    models['unified'] = {
        'model': rf_unified,
        'scaler': scaler
    }
    
    results['unified'] = {
        'mse': unified_mse,
        'r2': unified_r2,
        'gear_accuracy': unified_gear_accuracy
    }
    
    # FIXED: Determine best model
    print(f"\n MODEL COMPARISON:")
    model_scores = {}
    for model_name, model_results in results.items():
        # Combined score: R² for controls + gear accuracy
        r2_metric = model_results.get('continuous_r2', model_results.get('r2', 0))
        gear_acc = model_results.get('gear_accuracy', 0)
        combined_score = (r2_metric * 0.7) + (gear_acc * 0.3)
        model_scores[model_name] = combined_score
        
        print(f"   {model_name}:")
        for metric, value in model_results.items():
            print(f"     {metric}: {value:.4f}")
        print(f"     combined_score: {combined_score:.4f}")
    
    best_model = max(model_scores, key=model_scores.get)
    print(f"\n BEST MODEL: {best_model} (score: {model_scores[best_model]:.4f})")
    
    return models, results, best_model

class TORCSAIDriver:
    """AI Driver class for TORCS"""
    
    def __init__(self, models, feature_columns, best_model):
        self.models = models
        self.feature_columns = feature_columns
        self.best_model = best_model
    
    def predict_controls(self, game_state):
        """Predict driving controls from current game state"""
        # Convert game state to feature vector
        feature_vector = []
        for feature in self.feature_columns:
            value = game_state.get(feature, 0.0)
            feature_vector.append(value)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Use the best model to predict
        if self.best_model == 'unified':
            model = self.models[self.best_model]['model']
            predictions = model.predict(feature_vector)[0]
            
            return {
                'steer': float(np.clip(predictions[0], -1.0, 1.0)),
                'accel': float(np.clip(predictions[1], 0.0, 1.0)),
                'brake': float(np.clip(predictions[2], 0.0, 1.0)),
                'gear': int(np.clip(np.round(predictions[3]), 1, 6))
            }
        else:
            # Use separate models
            continuous_model = self.models[self.best_model]['continuous']
            discrete_model = self.models[self.best_model]['discrete']
            
            # Scale features if using neural network
            if 'scaler' in self.models[self.best_model]:
                feature_vector_scaled = self.models[self.best_model]['scaler'].transform(feature_vector)
                continuous_pred = continuous_model.predict(feature_vector_scaled)[0]
                gear_pred = discrete_model.predict(feature_vector_scaled)[0]
            else:
                continuous_pred = continuous_model.predict(feature_vector)[0]
                gear_pred = discrete_model.predict(feature_vector)[0]
            
            return {
                'steer': float(np.clip(continuous_pred[0], -1.0, 1.0)),
                'accel': float(np.clip(continuous_pred[1], 0.0, 1.0)),
                'brake': float(np.clip(continuous_pred[2], 0.0, 1.0)),
                'gear': int(gear_pred)
            }

def save_complete_model(models, results, feature_columns, continuous_targets, discrete_targets, best_model):
    """Save all trained models and metadata"""
    print("\n SAVING COMPLETE MODEL")
    print("=" * 40)
    
    os.makedirs('trained_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save all models
    for model_name, model_data in models.items():
        model_path = f'trained_models/torcs_driver_{model_name}_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        print(f" Saved {model_name} model to: {model_path}")
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'continuous_targets': continuous_targets,
        'discrete_targets': discrete_targets,
        'best_model': best_model,
        'results': results,
        'timestamp': timestamp
    }
    
    metadata_path = f'trained_models/torcs_driver_metadata_{timestamp}.pkl'
    joblib.dump(metadata, metadata_path)
    print(f" Saved metadata to: {metadata_path}")
    
    # Create AI driver
    ai_driver = TORCSAIDriver(models, feature_columns, best_model)
    prediction_path = f'trained_models/torcs_driver_predictor_{timestamp}.pkl'
    joblib.dump(ai_driver, prediction_path)
    print(f" Saved AI driver to: {prediction_path}")
    
    return timestamp

def main():
    """Main training pipeline"""
    print(" TORCS AI DRIVER TRAINING PIPELINE")
    print("=" * 60)
    print(" Training AI to mimic human driving with 4 control outputs:")
    print("    Steering (-1 to 1)")
    print("    Acceleration (0 to 1)")  
    print("    Braking (0 to 1)")
    print("     Gear (1 to 6)")
    print(" AUTOMATICALLY filters out poor quality data (oval tracks, etc.)")
    print("=" * 60)
    
    # Load data
    data = load_and_combine_data()
    if data is None:
        print(" ERROR: Could not load quality training data!")
        return None
    
    # Preprocess
    data = preprocess_data(data)
    if data is None or len(data) < 1000:
        print(" ERROR: Not enough clean data for training!")
        return None
    
    # Prepare features
    X, y_continuous, y_discrete, feature_columns, continuous_targets, discrete_targets = prepare_features_and_targets(data)
    
    # Split data
    print(f"\n SPLITTING DATA FOR TRAINING")
    print("=" * 40)
    X_train, X_val, y_continuous_train, y_continuous_val = train_test_split(
        X, y_continuous, test_size=0.2, random_state=42
    )
    _, _, y_discrete_train, y_discrete_val = train_test_split(
        X, y_discrete, test_size=0.2, random_state=42
    )
    
    print(f" Training samples: {len(X_train)}")
    print(f" Validation samples: {len(X_val)}")
    
    # Train models
    models, results, best_model = train_driving_models(
        X_train, X_val, y_continuous_train, y_continuous_val,
        y_discrete_train, y_discrete_val
    )
    
    # Save everything
    timestamp = save_complete_model(models, results, feature_columns, 
                                   continuous_targets, discrete_targets, best_model)
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f" Best performing model: {best_model}")
    print(f" Training timestamp: {timestamp}")
    
    print(f"\n MODEL PERFORMANCE SUMMARY:")
    best_results = results[best_model]
    gear_acc = best_results.get('gear_accuracy', 0)
    control_r2 = best_results.get('continuous_r2', best_results.get('r2', 0))
    
    print(f"     Gear Prediction: {gear_acc:.1%} accuracy")
    print(f"    Continuous Controls: R² = {control_r2:.3f}")
    
    if gear_acc > 0.9 and control_r2 > 0.6:
        print("    EXCELLENT: Model ready for professional racing!")
    elif gear_acc > 0.8 and control_r2 > 0.4:
        print("    GOOD: Model ready for autonomous driving!")
    else:
        print("     FAIR: Model functional but may need more training data")
    
    print(f"\n Files created:")
    print(f"    trained_models/torcs_driver_*_{timestamp}.pkl (model files)")
    print(f"    trained_models/torcs_driver_metadata_{timestamp}.pkl (metadata)")
    print(f"    trained_models/torcs_driver_predictor_{timestamp}.pkl (AI driver class)")
    
    print(f"\n NEXT STEPS:")
    print(f"   1. Load AI driver: joblib.load('trained_models/torcs_driver_predictor_{timestamp}.pkl')")
    print(f"   2. Integrate into your TORCS driver code")
    print(f"   3. Test in TORCS - it should fix the right turn bias!")
    print(f"   4. The AI learned YOUR driving style - race away! 🏎️")
    
    return models, results, timestamp

if __name__ == "__main__":
    if not os.path.exists('telemetry_data'):
        print(" Creating 'telemetry_data' directory...")
        os.makedirs('telemetry_data')
        print(" SETUP INSTRUCTIONS:")
        print("   1. Place your CSV telemetry files in the 'telemetry_data' directory")
        print("   2. Run this script again: python train_model_fixed.py")
    else:
        try:
            models, results, timestamp = main()
            if models:
                print(f"\n SUCCESS! Your AI driver is ready!")
                print(f" Load it with: joblib.load('trained_models/torcs_driver_predictor_{timestamp}.pkl')")
        except Exception as e:
            print(f"\n TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()