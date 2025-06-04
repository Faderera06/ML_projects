import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define Column Names 
feature_names = [
    "unnamed_idx", # <-- ADDED THIS DUMMY COLUMN NAME
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty" # 'attack_type' is the target
]

# Loading the Dataset
try:
    
    df_train = pd.read_csv('KDDTrain+.txt', names=feature_names, sep=',')
    df_test = pd.read_csv('KDDTest+.txt', names=feature_names, sep=',')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the same directory as the script.")
    print("You can download them from: https://www.kaggle.com/datasets/hassan06/nslkdd")
    exit()

# Combining for consistent preprocessing, then split again
df_combined = pd.concat([df_train.drop(columns=['difficulty']), df_test.drop(columns=['difficulty'])], ignore_index=True)

# NEW: Drop the 'unnamed_idx' column if it exists 
# This accounts for the common scenario where the first column is an unwanted index.
if 'unnamed_idx' in df_combined.columns:
    df_combined = df_combined.drop(columns=['unnamed_idx'])
    print("Dropped 'unnamed_idx' column.")

#  Preprocessing
# Identify categorical and numerical features
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = [col for col in df_combined.columns if col not in categorical_features and col != 'attack_type']

# Apply Label Encoding to categorical features
for col in categorical_features:
    le = LabelEncoder()
    
    df_combined[col] = df_combined[col].astype(str)
    df_combined[col] = le.fit_transform(df_combined[col])
    print(f"Label encoded '{col}'")

# Scale numerical features
scaler = StandardScaler()

for col in numerical_features:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

#  MODIFIED: Fill NaNs instead of dropping rows 
df_combined[numerical_features] = df_combined[numerical_features].fillna(df_combined[numerical_features].mean())
print("Filled NaN values in numerical features with column means.")


df_combined[numerical_features] = scaler.fit_transform(df_combined[numerical_features])
print("Numerical features scaled.")

# Convert attack_type to binary: 'normal' vs 'anomaly'

df_combined['is_anomaly'] = df_combined['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

# Separate features (X) and target (y)
X = df_combined.drop(columns=['attack_type', 'is_anomaly'])
y = df_combined['is_anomaly']

# Train Isolation Forest Model 
# contamination: The proportion of outliers in the data set.


contamination_rate = y.sum() / len(y) # Proportion of actual anomalies
print(f"Estimated anomaly contamination rate: {contamination_rate:.4f}")

# Initialize Isolation Forest
# random_state for reproducibility
model = IsolationForest(
    contamination=contamination_rate,
    random_state=42,
    n_estimators=100, # Number of base estimators (trees)
    max_features=1.0, # Number of features to consider when building a tree
    max_samples='auto' # Number of samples to draw from X to train each base estimator
)

# Fit the model (unsupervised learning, so no 'y' needed for fit)
model.fit(X)
print("Isolation Forest model trained.")

# Predict anomalies: -1 for outliers, 1 for inliers

predictions = model.predict(X)

# Converted Isolation Forest output (-1, 1) to our binary (0, 1) for evaluation
# -1 (outlier) -> 1 (anomaly)
# 1 (inlier) -> 0 (normal)
predicted_anomalies = np.where(predictions == -1, 1, 0)

#  Evaluate the Model 
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y, predicted_anomalies):.4f}")
print("\nClassification Report:")
print(classification_report(y, predicted_anomalies, target_names=['Normal', 'Anomaly']))

# Demonstrated a few predictions 
print("\n--- Sample Predictions ---")
# Selected a few representative indices that are guaranteed to exist
# picked the first 5 indices from the valid_indices list,
# or fewer if the list itself has less than 5 elements.
valid_indices = df_combined.index.tolist() # Get the current DataFrame indices

sample_indices_to_display = valid_indices[:min(len(valid_indices), 5)] # Take the first 5 or fewer

if not sample_indices_to_display:
    print("No valid samples to display after preprocessing.")
else:
    for idx in sample_indices_to_display:
        # Use .loc for Series indexing (y) and get_loc for array indexing (predicted_anomalies)
        actual_label = 'Anomaly' if y.loc[idx] == 1 else 'Normal'
        predicted_label = 'Anomaly' if predicted_anomalies[df_combined.index.get_loc(idx)] == 1 else 'Normal'
        print(f"Index {idx}: Actual = {actual_label}, Predicted = {predicted_label}")

print("\nProject 1 (Network Intrusion Detection) completed successfully.")
