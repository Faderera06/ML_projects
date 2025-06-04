import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM # One-Class SVM for anomaly detection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

    # Generate Synthetic Multivariate XR Telemetry Data 
print("Generating synthetic multivariate XR telemetry data...")
np.random.seed(42)

    # Features: 'head_rot_speed', 'hand_dist_moved', 'gaze_dwell_time_ms', 'interaction_rate'
num_normal_samples = 900
num_anomalies = 30 # A small percentage of outliers

    # Generate normal data: behaviors within expected ranges for a typical user
normal_data = pd.DataFrame({
        'head_rot_speed': np.random.normal(loc=1.5, scale=0.3, size=num_normal_samples), # rad/s
        'hand_dist_moved': np.random.normal(loc=0.8, scale=0.2, size=num_normal_samples), # meters/s
        'gaze_dwell_time_ms': np.random.normal(loc=500, scale=100, size=num_normal_samples).astype(int),
        'interaction_rate': np.random.normal(loc=2.0, scale=0.5, size=num_normal_samples).clip(0, 5) # per second
    })

    # Generate anomalous data: unusual combinations or extreme values
    # e.g., very high hand movement with low head rotation (teleportation glitch/manipulation)
    # or very high interaction rate with very short gaze dwell time (bot-like activity)
anomaly_data = pd.DataFrame({
        'head_rot_speed': np.random.normal(loc=0.1, scale=0.05, size=num_anomalies), # Abnormally low head speed
        'hand_dist_moved': np.random.normal(loc=5.0, scale=1.0, size=num_anomalies), # Abnormally high hand movement
        'gaze_dwell_time_ms': np.random.normal(loc=50, scale=20, size=num_anomalies).astype(int), # Abnormally low gaze time
        'interaction_rate': np.random.normal(loc=8.0, scale=1.0, size=num_anomalies).clip(5, 10) # Abnormally high interaction
    })

    # Combine data and shuffle
df = pd.concat([normal_data, anomaly_data], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Generated {len(df)} total XR telemetry records ({num_anomalies} synthetic anomalies).")

    # Features for OCSVM
X = df.copy()

    # 2. Preprocessing: Scale Features 
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Train One-Class SVM Model
    # One-Class SVM is an unsupervised algorithm used for outlier detection.
    # It learns a decision boundary that encapsulates the 'normal' data points,
    # classifying anything outside this boundary as an anomaly.
    # `nu` parameter is an upper bound on the fraction of training errors
    # and a lower bound of the fraction of support vectors. It should be (0, 1].
    # For anomaly detection, it often represents the expected proportion of outliers.
print("Training One-Class SVM model...")
model = OneClassSVM(kernel='rbf', nu=0.03, gamma='auto') # nu=0.03 implies expecting ~3% anomalies
model.fit(X_scaled_df)

    # Predict anomalies (-1 for outlier, 1 for inlier)
df['anomaly_prediction'] = model.predict(X_scaled_df)
df['is_anomaly'] = df['anomaly_prediction'].apply(lambda x: 1 if x == -1 else 0) # Convert to 0/1

detected_anomalies_count = df['is_anomaly'].sum()
print(f"Detected {detected_anomalies_count} anomalies.")

    # Visualize Results (using two features for 2D plot) 
print("Generating visualization (using head rotation speed vs hand distance moved)...")
plt.figure(figsize=(10, 7))
sns.scatterplot(
        x='head_rot_speed',
        y='hand_dist_moved',
        hue='is_anomaly',
        palette={0: 'blue', 1: 'red'},
        data=df,
        s=100,
        alpha=0.7,
        style='is_anomaly',
        markers={0: 'o', 1: 'X'}
    )
plt.title('Multivariate Anomaly Detection (One-Class SVM)')
plt.xlabel('Head Rotation Speed (rad/s)')
plt.ylabel('Hand Distance Moved (meters/s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'])
plt.tight_layout()
plt.savefig('multivariate_anomaly_plot.png') # Save the plot
plt.show()

print("\nProject 7 (Multivariate Anomaly Detection) completed.")
    