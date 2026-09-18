# Usage Instructions for PyTorch Regression Model

## Model Information
- **Model Name**: AutoEncoder_Net_gaussian_noise_light_MinMaxScaler
- **Architecture**: AutoEncoder_Net
- **Augmentation**: gaussian_noise_light
- **Scaler**: MinMaxScaler
- **Features**: ['PH', 'Cond', 'Salinity', 'ORP', 'Biomass', 'FishNum', 'Calendar', 'UV', 'AirPressure(11:00)', 'AirTemperature(11:00)', 'RelativeHumidity(11:00)']
- **Targets**: ['NH3']
- **Performance**:
  - MAE: 0.0655
  - RMSE: 0.0910
  - R²: 0.9696
  - Stopped at Epoch: 1925

## Loading and Using the Model

```python
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Load model package
with open('C:/Users/LBK/Downloads/6x AUG v22\AutoEncoder_Net_gaussian_noise_light_MinMaxScaler.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
scaler = model_package['scaler']
feature_names = model_package['feature_names']
target_names = model_package['target_names']

# Define model architecture (copy from training code)
class AutoEncoder_Net(nn.Module):
    # ... (copy architecture definition from main code)
    pass

# Reconstruct and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder_Net(
    model_package['input_dim'],
    model_package['output_dim']
).to(device)
model.load_state_dict(model_package['model_state_dict'])
model.eval()

# Prediction function
def predict_new_data(new_data):
    """
    Make predictions on new data
    
    Parameters:
    - new_data: numpy array or DataFrame with features: ['PH', 'Cond', 'Salinity', 'ORP', 'Biomass', 'FishNum', 'Calendar', 'UV', 'AirPressure(11:00)', 'AirTemperature(11:00)', 'RelativeHumidity(11:00)']
    
    Returns:
    - predictions: numpy array of predicted values
    """
    # Convert DataFrame to array
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data[feature_names].values
    
    # Ensure correct shape
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(1, -1)
    
    # Scale data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(new_data_scaled).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    return predictions

# Example usage
# Single prediction
new_sample = np.array([[value1, value2, ...]])  # Replace with actual values
prediction = predict_new_data(new_sample)
print(f"Prediction: {prediction}")

# DataFrame prediction
new_df = pd.DataFrame({
    'PH': [val1, val2, ...],
    # ... add all features
})
predictions = predict_new_data(new_df)
print(f"Predictions: {predictions}")
```

## Requirements
- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn

## Notes
- Ensure input features match training data order
- Handle missing values before prediction
- Model uses cuda for computation
- Early stopping patience: Check training logs for optimal configuration
