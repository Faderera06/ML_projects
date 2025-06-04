import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor # Good for density-based anomaly detection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Synthetic XR User Behavior Data


np.random.seed(42) # for reproducibility

num_normal_samples = 1000
num_anomaly_samples = 50

# Normal behavior: relatively consistent speed, low turn rate, moderate interaction
normal_data = pd.DataFrame({
    'avg_speed': np.random.normal(0.5, 0.1, num_normal_samples),
    'turn_rate': np.random.normal(0.1, 0.05, num_normal_samples),
    'interaction_frequency': np.random.normal(0.8, 0.2, num_normal_samples)
})
# Ensure values are non-negative
normal_data = normal_data.clip(lower=0)

# Anomaly behavior: sudden high speed, erratic turns, very low or very high interaction
anomaly_data = pd.DataFrame({
    'avg_speed': np.random.normal(2.0, 0.5, num_anomaly_samples), # Much faster
    'turn_rate': np.random.normal(1.5, 0.7, num_anomaly_samples), # Erratic turns
    'interaction_frequency': np.random.choice([0.1, 3.0], num_anomaly_samples) # Very low or very high
})
anomaly_data = anomaly_data.clip(lower=0)

# Combine data and label (0 for normal, 1 for anomaly)
data = pd.concat([normal_data, anomaly_data], ignore_index=True)
labels = np.array([0] * num_normal_samples + [1] * num_anomaly_samples)

print(f"Generated {num_normal_samples} normal samples and {num_anomaly_samples} anomaly samples.")

# Preprocessing 
# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("Data scaled.")

# Train Local Outlier Factor (LOF) Model

contamination_rate = num_anomaly_samples / (num_normal_samples + num_anomaly_samples)
print(f"Estimated anomaly contamination rate: {contamination_rate:.4f}")

model = LocalOutlierFactor(
    n_neighbors=20, # Number of neighbors to consider
    contamination=contamination_rate,
    novelty=True # Set to True for consistent prediction on new data
)

# Fit the model (unsupervised)
model.fit(scaled_data)
print("Local Outlier Factor model trained.")

# Predict anomalies: -1 for outliers, 1 for inliers

predictions = model.predict(scaled_data)

# Convert LOF output (-1, 1) to our binary (0, 1) for evaluation
predicted_anomalies = np.where(predictions == -1, 1, 0)

# Evaluate the Model 
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(labels, predicted_anomalies):.4f}")
print("\nClassification Report:")
print(classification_report(labels, predicted_anomalies, target_names=['Normal', 'Anomaly']))

# Visualize Results 

plt.figure(figsize=(10, 7))
sns.scatterplot(x=data['avg_speed'], y=data['turn_rate'], hue=labels, style=predicted_anomalies,
                palette={0: 'blue', 1: 'red'}, markers={0: 'o', 1: 'X'}, s=100, alpha=0.7)
plt.title('LOF Anomaly Detection on Simulated XR User Behavior')
plt.xlabel('Average Speed')
plt.ylabel('Turn Rate')
plt.legend(title='Actual/Predicted')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nProject 2 (Simulated XR User Behavior Anomaly Detection) completed successfully.")
