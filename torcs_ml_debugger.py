import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TORCSMLDebugger:
    """Comprehensive debugging tool for TORCS ML driving models"""
    
    def __init__(self, model_path=None, data_path=None, metadata_path=None):
        self.model = None
        self.metadata = None
        self.data = None
        self.feature_names = None
        
        print("🔧 TORCS ML Model Debugger v1.0")
        print("=" * 50)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.auto_find_model()
        
        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            self.load_metadata(metadata_path)
        else:
            self.auto_find_metadata()
        
        # Load training data
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
        
        print("🚀 Debugger ready! Use debug_all() for complete analysis")
    
    def auto_find_model(self):
        """Automatically find the latest model file"""
        model_dir = 'trained_models'
        if os.path.exists(model_dir):
            rf_files = [f for f in os.listdir(model_dir) if 'random_forest' in f and f.endswith('.pkl')]
            if rf_files:
                latest_model = sorted(rf_files)[-1]
                model_path = os.path.join(model_dir, latest_model)
                self.load_model(model_path)
                print(f"📂 Auto-loaded model: {latest_model}")
            else:
                print("❌ No model files found in trained_models/")
        else:
            print("❌ trained_models directory not found")
    
    def auto_find_metadata(self):
        """Automatically find the latest metadata file"""
        model_dir = 'trained_models'
        if os.path.exists(model_dir):
            metadata_files = [f for f in os.listdir(model_dir) if 'metadata' in f and f.endswith('.pkl')]
            if metadata_files:
                latest_metadata = sorted(metadata_files)[-1]
                metadata_path = os.path.join(model_dir, latest_metadata)
                self.load_metadata(metadata_path)
                print(f"📂 Auto-loaded metadata: {latest_metadata}")
    
    def load_model(self, model_path):
        """Load the ML model"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Determine model structure
            if hasattr(self.model, 'keys') and isinstance(self.model, dict):
                print(f"📊 Model structure: {list(self.model.keys())}")
            else:
                print("📊 Model structure: Unified model")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def load_metadata(self, metadata_path):
        """Load model metadata"""
        try:
            self.metadata = joblib.load(metadata_path)
            if 'feature_columns' in self.metadata:
                self.feature_names = self.metadata['feature_columns']
                print(f"✅ Metadata loaded: {len(self.feature_names)} features")
            else:
                print("⚠️  Metadata loaded but no feature_columns found")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    
    def load_data(self, data_path):
        """Load training data for analysis"""
        try:
            if data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.endswith('.pkl'):
                self.data = pd.read_pickle(data_path)
            else:
                print("❌ Unsupported data format. Use .csv or .pkl")
                return
            
            print(f"✅ Training data loaded: {len(self.data)} samples")
            print(f"📊 Columns: {list(self.data.columns)}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def debug_all(self):
        """Run all debugging analyses"""
        print("\n" + "="*60)
        print("🔍 STARTING COMPREHENSIVE ML MODEL DEBUG")
        print("="*60)
        
        if self.model is None:
            print("❌ No model loaded. Cannot proceed with debugging.")
            return
        
        # Run all debug functions
        self.analyze_feature_importance()
        self.test_model_logic()
        self.analyze_sensor_relationships()
        
        if self.data is not None:
            self.analyze_training_data()
            self.test_model_predictions()
        
        self.generate_debug_report()
    
    def analyze_feature_importance(self):
        """Analyze which features the model considers most important"""
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        try:
            # Handle different model structures
            importance_data = None
            
            if hasattr(self.model, 'keys') and isinstance(self.model, dict):
                if 'continuous' in self.model and hasattr(self.model['continuous'], 'feature_importances_'):
                    importance_data = self.model['continuous'].feature_importances_
                elif hasattr(self.model, 'feature_importances_'):
                    importance_data = self.model.feature_importances_
            elif hasattr(self.model, 'feature_importances_'):
                importance_data = self.model.feature_importances_
            
            if importance_data is None:
                print("❌ Model doesn't support feature importance analysis")
                return
            
            if self.feature_names is None:
                print("❌ No feature names available")
                return
            
            # Create importance ranking
            feature_importance = list(zip(self.feature_names, importance_data))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("TOP 15 MOST IMPORTANT FEATURES:")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"{i+1:2d}. {feature:<25} : {importance:.4f}")
            
            # Analyze sensor importance patterns
            sensor_importance = {}
            other_importance = {}
            
            for feature, importance in feature_importance:
                if 'track_sensor_' in feature:
                    angle = int(feature.split('_')[-1])
                    sensor_importance[angle] = importance
                else:
                    other_importance[feature] = importance
            
            print(f"\n📡 SENSOR IMPORTANCE BY DIRECTION:")
            left_importance = sum(imp for angle, imp in sensor_importance.items() if angle < 0)
            center_importance = sensor_importance.get(0, 0)
            right_importance = sum(imp for angle, imp in sensor_importance.items() if angle > 0)
            
            print(f"Left sensors (-90° to -10°):  {left_importance:.4f}")
            print(f"Center sensor (0°):          {center_importance:.4f}")
            print(f"Right sensors (10° to 90°):  {right_importance:.4f}")
            
            # Check for balance
            if abs(left_importance - right_importance) > 0.1:
                print(f"⚠️  WARNING: Unbalanced left/right sensor importance!")
                print(f"   Difference: {abs(left_importance - right_importance):.4f}")
            else:
                print("✅ Left/right sensor importance is balanced")
            
            # Top non-sensor features
            print(f"\nTOP NON-SENSOR FEATURES:")
            non_sensor_sorted = sorted(other_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in non_sensor_sorted[:5]:
                print(f"   {feature:<20} : {importance:.4f}")
                
        except Exception as e:
            print(f"❌ Error in feature importance analysis: {e}")
    
    def test_model_logic(self):
        """Test model with synthetic scenarios to validate logic"""
        print("\n🧠 MODEL LOGIC TESTING")
        print("-" * 40)
        
        if self.feature_names is None:
            print("❌ No feature names available for logic testing")
            return
        
        # Create baseline scenario
        baseline_features = {
            'speed_x': 50.0, 'speed_y': 0.0, 'speed_z': 0.0,
            'angle': 0.0, 'track_position': 0.0, 'rpm': 3000.0,
            'fuel': 50.0, 'damage': 0.0, 'distance_from_start': 1000.0,
            'gear': 3,
            'wheel_speed_front_left': 50.0, 'wheel_speed_front_right': 50.0,
            'wheel_speed_rear_left': 50.0, 'wheel_speed_rear_right': 50.0
        }
        
        # Set all sensors to open road
        for angle in range(-90, 100, 10):
            baseline_features[f'track_sensor_{angle}'] = 200.0
        
        def create_test_vector(modifications):
            features = baseline_features.copy()
            features.update(modifications)
            return np.array([features.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)
        
        def get_prediction(test_vector):
            try:
                if hasattr(self.model, 'keys') and isinstance(self.model, dict):
                    if 'continuous' in self.model:
                        prediction = self.model['continuous'].predict(test_vector)[0]
                    else:
                        prediction = list(self.model.values())[0].predict(test_vector)[0]
                else:
                    prediction = self.model.predict(test_vector)[0]
                
                # Handle different prediction formats
                if len(prediction) >= 3:
                    return prediction[0], prediction[1], prediction[2]  # steer, accel, brake
                else:
                    return prediction[0], 0.5, 0.0  # Only steering available
            except Exception as e:
                print(f"   ❌ Prediction error: {e}")
                return 0.0, 0.0, 0.0
        
        # Test scenarios
        test_scenarios = [
            ("🏁 Baseline (open road)", {}),
            ("🧱 Wall on LEFT side", {
                'track_sensor_-30': 15.0, 'track_sensor_-20': 10.0, 'track_sensor_-10': 20.0
            }),
            ("🧱 Wall on RIGHT side", {
                'track_sensor_30': 15.0, 'track_sensor_20': 10.0, 'track_sensor_10': 20.0
            }),
            ("🚧 Wall straight AHEAD", {
                'track_sensor_0': 15.0, 'track_sensor_-10': 30.0, 'track_sensor_10': 30.0
            }),
            ("↩️  Sharp LEFT turn", {
                'track_sensor_10': 30.0, 'track_sensor_20': 20.0, 'track_sensor_30': 10.0,
                'track_sensor_-10': 150.0, 'track_sensor_-20': 180.0
            }),
            ("↪️  Sharp RIGHT turn", {
                'track_sensor_-10': 30.0, 'track_sensor_-20': 20.0, 'track_sensor_-30': 10.0,
                'track_sensor_10': 150.0, 'track_sensor_20': 180.0
            }),
            ("⬅️ Car off-track LEFT", {'track_position': -0.8}),
            ("➡️ Car off-track RIGHT", {'track_position': 0.8}),
            ("🔄 Car angled LEFT", {'angle': -0.5}),
            ("🔄 Car angled RIGHT", {'angle': 0.5}),
        ]
        
        print("SCENARIO TESTING:")
        logic_results = []
        
        for emoji_name, modifications in test_scenarios:
            test_vector = create_test_vector(modifications)
            steer, accel, brake = get_prediction(test_vector)
            
            # Determine steering direction
            if steer > 0.1:
                steer_dir = "RIGHT"
            elif steer < -0.1:
                steer_dir = "LEFT"
            else:
                steer_dir = "STRAIGHT"
            
            print(f"{emoji_name:<25}: Steer={steer:6.3f} ({steer_dir}) | Accel={accel:.2f} | Brake={brake:.2f}")
            logic_results.append((emoji_name, steer, accel, brake))
        
        # Logic validation
        print(f"\n🔍 LOGIC VALIDATION:")
        validations = [
            ("Wall on LEFT → should steer RIGHT", logic_results[1][1] > 0.1),
            ("Wall on RIGHT → should steer LEFT", logic_results[2][1] < -0.1),
            ("Wall AHEAD → should brake", logic_results[3][3] > 0.2),
            ("LEFT turn → should steer LEFT", logic_results[4][1] < -0.1),
            ("RIGHT turn → should steer RIGHT", logic_results[5][1] > 0.1),
            ("Off-track LEFT → should steer RIGHT", logic_results[6][1] > 0.1),
            ("Off-track RIGHT → should steer LEFT", logic_results[7][1] < -0.1),
        ]
        
        passed_tests = 0
        for test_name, passed in validations:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name}")
            if passed:
                passed_tests += 1
        
        print(f"\n📊 LOGIC TEST SCORE: {passed_tests}/{len(validations)} ({passed_tests/len(validations)*100:.1f}%)")
        
        if passed_tests < len(validations) * 0.7:
            print("⚠️  WARNING: Model failed basic logic tests! Consider retraining.")
        elif passed_tests == len(validations):
            print("🎉 EXCELLENT: Model passed all logic tests!")
        else:
            print("👍 GOOD: Model passed most logic tests.")
    
    def analyze_sensor_relationships(self):
        """Analyze spatial relationships between sensors"""
        print("\n🔗 SENSOR RELATIONSHIP ANALYSIS")
        print("-" * 40)
        
        if self.feature_names is None:
            print("❌ No feature names available")
            return
        
        # Test sensor gradient understanding
        def test_sensor_gradient(center_angle, adjacent_angles, wall_distance):
            base_features = {f'track_sensor_{i}': 200.0 for i in range(-90, 100, 10)}
            base_features.update({
                'speed_x': 50.0, 'angle': 0.0, 'track_position': 0.0, 'gear': 3,
                'rpm': 3000.0, 'fuel': 50.0, 'damage': 0.0, 'distance_from_start': 1000.0
            })
            
            # Set wall pattern
            base_features[f'track_sensor_{center_angle}'] = wall_distance
            for angle in adjacent_angles:
                if f'track_sensor_{angle}' in base_features:
                    base_features[f'track_sensor_{angle}'] = wall_distance + 20.0
            
            feature_vector = np.array([base_features.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)
            
            try:
                if hasattr(self.model, 'keys') and isinstance(self.model, dict):
                    if 'continuous' in self.model:
                        prediction = self.model['continuous'].predict(feature_vector)[0]
                    else:
                        prediction = list(self.model.values())[0].predict(feature_vector)[0]
                else:
                    prediction = self.model.predict(feature_vector)[0]
                
                return prediction[0] if len(prediction) > 0 else 0.0
            except:
                return 0.0
        
        # Test different sensor configurations
        print("TESTING SENSOR SPATIAL RELATIONSHIPS:")
        
        tests = [
            ("Left sensor cluster", -20, [-30, -10], "Should cause RIGHT steering"),
            ("Right sensor cluster", 20, [10, 30], "Should cause LEFT steering"),
            ("Front-left cluster", -10, [-20, 0], "Should cause slight RIGHT steering"),
            ("Front-right cluster", 10, [0, 20], "Should cause slight LEFT steering"),
        ]
        
        relationship_score = 0
        for test_name, center, adjacent, expected in tests:
            steering = test_sensor_gradient(center, adjacent, 20.0)
            
            # Determine if response is correct
            if "RIGHT" in expected and steering > 0.05:
                correct = True
            elif "LEFT" in expected and steering < -0.05:
                correct = True
            else:
                correct = False
            
            status = "✅" if correct else "❌"
            print(f"   {status} {test_name:<20}: Steer={steering:6.3f} | {expected}")
            
            if correct:
                relationship_score += 1
        
        print(f"\n📊 RELATIONSHIP SCORE: {relationship_score}/{len(tests)} ({relationship_score/len(tests)*100:.1f}%)")
        
        if relationship_score < len(tests) * 0.5:
            print("⚠️  WARNING: Model doesn't understand sensor spatial relationships!")
            print("   💡 Solution: Add more diverse training data with various wall configurations")
    
    def analyze_training_data(self):
        """Analyze the training data for bias and balance"""
        print("\n📊 TRAINING DATA ANALYSIS")
        print("-" * 40)
        
        if self.data is None:
            print("❌ No training data loaded")
            return
        
        # Check if required columns exist
        required_cols = ['steer']
        if not all(col in self.data.columns for col in required_cols):
            print(f"❌ Missing required columns. Found: {list(self.data.columns)}")
            return
        
        # Steering distribution analysis
        print("STEERING DISTRIBUTION:")
        left_count = len(self.data[self.data['steer'] < -0.1])
        straight_count = len(self.data[abs(self.data['steer']) <= 0.1])
        right_count = len(self.data[self.data['steer'] > 0.1])
        total = len(self.data)
        
        print(f"   Left turns  (< -0.1): {left_count:6d} ({left_count/total*100:5.1f}%)")
        print(f"   Straight    (±0.1):   {straight_count:6d} ({straight_count/total*100:5.1f}%)")
        print(f"   Right turns (> 0.1):  {right_count:6d} ({right_count/total*100:5.1f}%)")
        print(f"   Total samples:        {total:6d}")
        
        # Calculate bias
        mean_steer = self.data['steer'].mean()
        std_steer = self.data['steer'].std()
        print(f"\n   Mean steering: {mean_steer:6.4f}")
        print(f"   Std deviation: {std_steer:6.4f}")
        
        # Bias analysis
        if abs(mean_steer) > 0.05:
            bias_direction = "RIGHT" if mean_steer > 0 else "LEFT"
            print(f"⚠️  WARNING: Training data has {bias_direction} bias! Mean = {mean_steer:.4f}")
            print(f"   💡 This explains why your model always turns right!")
        else:
            print("✅ Training data steering is well balanced")
        
        # Balance analysis
        left_right_ratio = left_count / max(right_count, 1)
        if left_right_ratio < 0.7 or left_right_ratio > 1.3:
            print(f"⚠️  WARNING: Unbalanced left/right turns! Ratio = {left_right_ratio:.2f}")
            print(f"   💡 Consider data augmentation (mirroring) to balance the dataset")
        else:
            print("✅ Left/right turn distribution is balanced")
        
        # Speed analysis
        if 'speed_x' in self.data.columns:
            print(f"\nSPEED DISTRIBUTION:")
            print(f"   Mean speed: {self.data['speed_x'].mean():6.1f} km/h")
            print(f"   Max speed:  {self.data['speed_x'].max():6.1f} km/h")
            print(f"   Min speed:  {self.data['speed_x'].min():6.1f} km/h")
        
        # Sensor analysis
        sensor_cols = [col for col in self.data.columns if 'track_sensor' in col]
        if sensor_cols:
            print(f"\nSENSOR DATA:")
            print(f"   Number of sensors: {len(sensor_cols)}")
            sensor_means = {col: self.data[col].mean() for col in sensor_cols}
            
            # Check for sensor bias
            left_sensors = [col for col in sensor_cols if 'track_sensor_-' in col]
            right_sensors = [col for col in sensor_cols if col.endswith(('10', '20', '30', '40', '50', '60', '70', '80', '90'))]
            
            if left_sensors and right_sensors:
                left_mean = np.mean([sensor_means[col] for col in left_sensors])
                right_mean = np.mean([sensor_means[col] for col in right_sensors])
                print(f"   Left sensors avg:  {left_mean:6.1f}m")
                print(f"   Right sensors avg: {right_mean:6.1f}m")
                
                if abs(left_mean - right_mean) > 20:
                    print(f"⚠️  WARNING: Unbalanced sensor data! Difference = {abs(left_mean - right_mean):.1f}m")
                else:
                    print("✅ Sensor data appears balanced")
    
    def test_model_predictions(self):
        """Test model predictions on training data"""
        print("\n🎯 MODEL PREDICTION TESTING")
        print("-" * 40)
        
        if self.data is None or self.model is None:
            print("❌ Need both model and training data for prediction testing")
            return
        
        try:
            # Prepare test data
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in self.data.columns]
                missing_features = [f for f in self.feature_names if f not in self.data.columns]
                
                if missing_features:
                    print(f"⚠️  Missing features in data: {missing_features}")
                
                X_test = self.data[available_features].fillna(0)
            else:
                print("❌ No feature names available")
                return
            
            # Make predictions
            if hasattr(self.model, 'keys') and isinstance(self.model, dict):
                if 'continuous' in self.model:
                    predictions = self.model['continuous'].predict(X_test)
                else:
                    predictions = list(self.model.values())[0].predict(X_test)
            else:
                predictions = self.model.predict(X_test)
            
            # Extract steering predictions
            if len(predictions.shape) > 1 and predictions.shape[1] > 0:
                steer_pred = predictions[:, 0]
            else:
                steer_pred = predictions
            
            steer_actual = self.data['steer'].values
            
            # Calculate metrics
            mse = mean_squared_error(steer_actual, steer_pred)
            r2 = r2_score(steer_actual, steer_pred)
            
            print(f"PREDICTION ACCURACY:")
            print(f"   Mean Squared Error: {mse:.6f}")
            print(f"   R² Score:           {r2:.6f}")
            
            # Analyze prediction bias
            pred_mean = np.mean(steer_pred)
            actual_mean = np.mean(steer_actual)
            bias = pred_mean - actual_mean
            
            print(f"\nPREDICTION BIAS:")
            print(f"   Actual mean steering:    {actual_mean:.6f}")
            print(f"   Predicted mean steering: {pred_mean:.6f}")
            print(f"   Bias (pred - actual):    {bias:.6f}")
            
            if abs(bias) > 0.05:
                bias_direction = "RIGHT" if bias > 0 else "LEFT"
                print(f"⚠️  WARNING: Model has {bias_direction} prediction bias!")
            else:
                print("✅ Model predictions are unbiased")
            
            # Direction accuracy
            actual_directions = np.where(steer_actual > 0.1, 1, np.where(steer_actual < -0.1, -1, 0))
            pred_directions = np.where(steer_pred > 0.1, 1, np.where(steer_pred < -0.1, -1, 0))
            direction_accuracy = np.mean(actual_directions == pred_directions)
            
            print(f"\nDIRECTION ACCURACY: {direction_accuracy:.3f} ({direction_accuracy*100:.1f}%)")
            
            if direction_accuracy < 0.7:
                print("❌ Poor direction accuracy - model needs retraining")
            elif direction_accuracy > 0.9:
                print("✅ Excellent direction accuracy")
            else:
                print("👍 Good direction accuracy")
                
        except Exception as e:
            print(f"❌ Error in prediction testing: {e}")
    
    def generate_debug_report(self):
        """Generate a summary debug report"""
        print("\n" + "="*60)
        print("📋 DEBUG SUMMARY REPORT")
        print("="*60)
        
        print("\n🔍 KEY FINDINGS:")
        print("   1. Check feature importance - are sensors properly weighted?")
        print("   2. Validate logic tests - does model respond correctly to walls?")
        print("   3. Examine training data bias - is steering data balanced?")
        print("   4. Review prediction accuracy - how well does model perform?")
        
        print("\n💡 COMMON SOLUTIONS:")
        print("   • RIGHT TURN BIAS → Balance training data or add data augmentation")
        print("   • POOR LOGIC → Add more diverse scenarios in training data")
        print("   • LOW ACCURACY → Try different model parameters or algorithms")
        print("   • SENSOR ISSUES → Check sensor feature engineering")
        
        print("\n🛠️  NEXT STEPS:")
        print("   1. Fix identified issues in training data")
        print("   2. Retrain model with improved data")
        print("   3. Re-run this debugger to verify improvements")
        print("   4. Test in actual TORCS environment")
        
        print("\n" + "="*60)
        print("🏁 DEBUG COMPLETE")
        print("="*60)

# Usage example
if __name__ == "__main__":
    # Initialize debugger
    debugger = TORCSMLDebugger()
    
    # Run complete debug analysis
    debugger.debug_all()
    
    # Or run individual analyses:
    # debugger.analyze_feature_importance()
    # debugger.test_model_logic()
    # debugger.analyze_sensor_relationships()
    
    # If you have training data:
    # debugger.load_data('path/to/your/training_data.csv')
    # debugger.analyze_training_data()
    # debugger.test_model_predictions()