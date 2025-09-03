# TORCS-AI-Racing-Driver
An advanced machine learning-based autonomous racing driver for TORCS (The Open Racing Car Simulator). This project implements a complete pipeline from human driving data collection to training sophisticated AI models that can race competitively on various tracks.

# TORCS AI Racing Driver

An advanced machine learning-based autonomous racing driver for TORCS (The Open Racing Car Simulator). This project implements a complete pipeline from human driving data collection to training sophisticated AI models that can race competitively on various tracks.

## Overview

This system uses pure machine learning approaches to create an AI driver that learns from human driving patterns. Unlike rule-based racing bots, this AI makes decisions based entirely on trained models, resulting in more natural and adaptive driving behavior.

## Key Features

### AI Driver Capabilities
- **Pure ML Control**: 100% machine learning-based decisions with no hardcoded rules
- **Multi-Output Prediction**: Simultaneous control of steering, acceleration, braking, and gear selection
- **Balanced Steering**: Addresses common AI racing issues like directional bias through improved training
- **Real-time Performance**: Optimized for live racing with minimal computational overhead

### Data Collection & Training
- **Human Telemetry Recording**: Captures complete driving sessions with all control inputs
- **Automatic Quality Filtering**: Detects and removes poor training data (oval tracks, stationary periods)
- **Multiple Model Architectures**: Implements Random Forest and Neural Network approaches
- **Comprehensive Validation**: Built-in logic testing and bias detection

### Development Tools
- **ML Debugger**: Advanced model analysis and performance diagnostics
- **Performance Monitoring**: Real-time statistics and success rate tracking
- **Feature Analysis**: Identifies important sensors and vehicle state variables
- **Training Visualization**: Data quality metrics and steering distribution analysis

## Project Structure

```
├── ai_driver.py           # Main AI racing driver
├── manual_driver.py       # Human-controlled driver for data collection
├── train_model.py         # Complete ML training pipeline
├── torcs_ml_debugger.py   # Model analysis and debugging tools
├── pyclient.py           # TORCS network client
├── carState.py           # Vehicle state management
├── carControl.py         # Vehicle control interface
├── msgParser.py          # TORCS message protocol handler
├── telemetry_data/       # Directory for training data
└── trained_models/       # Directory for saved models
```

## Installation

### Prerequisites
- Python 3.7 or higher
- TORCS racing simulator
- Required Python packages:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn joblib keyboard
```

### Setup
1. Clone this repository
2. Install dependencies
3. Ensure TORCS is installed and configured
4. Create data directories:
```bash
mkdir telemetry_data trained_models
```

## Usage

### 1. Collect Training Data
Record human driving sessions to create training datasets:

```bash
python pyclient.py --manual-transmission
```

Use arrow keys for steering and throttle/brake controls. The system automatically records telemetry data including all control inputs.

### 2. Train AI Models
Process collected data and train machine learning models:

```bash
python train_model.py
```

The training script automatically:
- Filters out poor quality data
- Trains multiple model architectures
- Validates model performance
- Selects the best performing model

### 3. Deploy AI Driver
Run the trained AI driver:

```bash
python pyclient.py
```

The AI will load the latest trained model and race autonomously.

### 4. Debug and Analyze
Use the debugging tools to analyze model performance:

```bash
python torcs_ml_debugger.py
```

## Model Architecture

The system implements several ML approaches:

- **Random Forest Regressor**: For continuous control outputs (steering, acceleration, braking)
- **Random Forest Classifier**: For discrete gear selection
- **Multi-layer Perceptron**: Neural network alternative for both continuous and discrete outputs
- **Unified Model**: Single model predicting all control outputs simultaneously

### Feature Engineering
The AI processes 30+ engineered features including:
- Vehicle dynamics (speed, position, orientation)
- Track sensors (19-point distance array)
- Engine parameters (RPM, gear, fuel)
- Wheel dynamics and physics simulation

## Performance Metrics

### Model Quality
- **Gear Accuracy**: >90% correct gear selection
- **Control R²**: >0.6 coefficient of determination for steering/throttle/brake
- **Steering Balance**: Left/right turn ratio within 0.5-2.0 range
- **Logic Validation**: Passes obstacle avoidance and racing line scenarios

### Racing Performance
- **Success Rate**: >95% prediction success during races
- **Speed Performance**: Maintains competitive lap times
- **Track Adaptation**: Handles various track layouts and conditions

## Technical Highlights

### Data Quality Assurance
- Automatic detection of oval track data (excessive straight-line driving)
- Filtering of stationary periods and invalid control inputs
- Balance analysis for left/right turn distribution
- Speed and distance validation

### Model Reliability
- Component-based model loading avoiding pickle serialization issues
- Robust error handling and fallback mechanisms
- Memory-efficient feature processing
- Real-time performance monitoring

### Network Communication
- UDP protocol handling for TORCS integration
- Connection timeout management and retry logic
- Message parsing and control output formatting

## Configuration

### Training Parameters
Key parameters can be adjusted in `train_model.py`:
- Model hyperparameters (n_estimators, max_depth, etc.)
- Data filtering thresholds
- Train/validation split ratios
- Feature selection criteria

### AI Driver Settings
Configure AI behavior in `ai_driver.py`:
- Prediction confidence thresholds
- Safety bounds for control outputs
- Performance reporting intervals

## Troubleshooting

### Common Issues
1. **Right Turn Bias**: Usually indicates unbalanced training data - use data augmentation
2. **Poor Logic Performance**: Add more diverse training scenarios
3. **Model Loading Errors**: Check file paths and model compatibility
4. **Network Timeouts**: Verify TORCS server configuration and network settings

### Debug Tools
Use `torcs_ml_debugger.py` to identify:
- Feature importance imbalances
- Training data bias patterns
- Model prediction accuracy
- Sensor relationship understanding

## Requirements

- **Operating System**: Linux, Windows, or macOS
- **Python**: 3.7+
- **TORCS**: Latest version with UDP server support
- **Memory**: 4GB+ recommended for model training
- **Storage**: 1GB+ for telemetry data and trained models

## Contributing

This project demonstrates practical machine learning applications in real-time control systems. Key areas for contribution include:
- Additional model architectures and algorithms
- Enhanced feature engineering techniques
- Improved data augmentation methods
- Advanced debugging and visualization tools

## License

This project is provided for educational and research purposes. Ensure compliance with TORCS licensing terms when using this code.

---

**Note**: This AI driver learns from human driving patterns and adapts to individual driving styles. Performance may vary based on training data quality and quantity. The system is designed for simulation environments and should not be adapted for real-world vehicle control without extensive additional safety measures.
