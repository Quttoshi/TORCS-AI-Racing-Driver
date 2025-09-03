# TORCS AI Racing Driver

Machine learning-based autonomous racing driver for TORCS that learns from human driving data to compete on various tracks.

## Features

- **Pure ML Control**: 100% data-driven decisions for steering, acceleration, braking, and gear selection
- **Human Data Collection**: Records telemetry from manual driving sessions
- **Quality Filtering**: Automatically removes poor training data (oval tracks, stationary periods)
- **Multiple Models**: Random Forest and Neural Network implementations
- **Debug Tools**: Performance analysis and bias detection

## Quick Start

1. **Collect Data**: Record human driving
   ```bash
   python pyclient.py --manual-transmission
   ```

2. **Train Models**: Process data and train AI
   ```bash
   python train_model.py
   ```

3. **Race**: Deploy trained AI driver
   ```bash
   python pyclient.py
   ```

## Project Structure

```
├── ai_driver.py           # AI racing driver
├── manual_driver.py       # Human driver for data collection
├── train_model.py         # ML training pipeline
├── torcs_ml_debugger.py   # Model analysis tools
├── pyclient.py           # TORCS client
└── telemetry_data/       # Training data directory
```

## Installation

```bash
pip install scikit-learn numpy pandas joblib keyboard
```

Requires Python 3.7+ and TORCS simulator.

## Technical Details

### Model Architecture
- **Random Forest**: Primary model for all control outputs
- **Neural Networks**: Alternative architecture option
- **Feature Engineering**: 30+ features including vehicle state and 19-sensor track array

### Performance Metrics
- Gear selection: >90% accuracy
- Control prediction: R² >0.6
- Steering balance: Left/right ratio 0.5-2.0
- Real-time success rate: >95%

### Key Components
- Automatic data quality assessment
- Component-based model loading
- Real-time performance monitoring
- Robust TORCS network communication

## Configuration

Training parameters in `train_model.py`:
- Model hyperparameters
- Data filtering thresholds
- Feature selection criteria

AI behavior in `ai_driver.py`:
- Control output bounds
- Performance reporting intervals

## Troubleshooting

Common issues and solutions:
- **Turn bias**: Balance training data or use data augmentation
- **Poor performance**: Add diverse training scenarios
- **Model errors**: Check file paths and compatibility
- **Network timeouts**: Verify TORCS server settings

Use `torcs_ml_debugger.py` for detailed model analysis.

## Requirements

- Python 3.7+
- TORCS with UDP support
- 4GB+ RAM (recommended for training)
- 1GB+ storage for data and models

---

This AI learns human driving patterns and adapts to individual styles. Designed for simulation environments only.
