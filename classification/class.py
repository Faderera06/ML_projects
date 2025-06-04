import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # A simple, interpretable classifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

    # --- 1. Generate Synthetic Security Alert Data ---
print("Generating synthetic security alert data...")
np.random.seed(42)

    # Features: 'packet_size', 'connection_duration', 'failed_attempts', 'source_reputation'
    # Target: 'is_malicious' (0 for normal, 1 for malicious)

num_normal_samples = 900
num_malicious_samples = 100 # Represents a smaller number of threats

    # Normal data: typical patterns
normal_data = pd.DataFrame({
        'packet_size': np.random.normal(loc=1500, scale=200, size=num_normal_samples).astype(int),
        'connection_duration': np.random.normal(loc=60, scale=20, size=num_normal_samples).astype(int),
        'failed_attempts': np.random.randint(0, 3, size=num_normal_samples),
        'source_reputation': np.random.normal(loc=0.9, scale=0.05, size=num_normal_samples) # High reputation
    })
normal_data['is_malicious'] = 0

    # Malicious data: atypical patterns
malicious_data = pd.DataFrame({
        'packet_size': np.random.normal(loc=50, scale=30, size=num_malicious_samples).astype(int), # Small, suspicious packets
        'connection_duration': np.random.normal(loc=5, scale=3, size=num_malicious_samples).astype(int), # Short, suspicious connections
        'failed_attempts': np.random.randint(5, 15, size=num_malicious_samples), # Many failed attempts
        'source_reputation': np.random.normal(loc=0.1, scale=0.05, size=num_malicious_samples) # Low reputation
    })
malicious_data['is_malicious'] = 1

    # Combine and shuffle data
df = pd.concat([normal_data, malicious_data], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Generated {len(df)} total alert records ({df['is_malicious'].sum()} malicious).")

    # Define features (X) and target (y)
X = df.drop('is_malicious', axis=1)
y = df['is_malicious']

    # --- 2. Preprocessing: Scale Numerical Features ---
print("Scaling numerical features...")
scaler = StandardScaler()
    # Scale all features
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # --- 3. Split Data into Training and Testing Sets ---
print("Splitting data into training and testing sets...")
    # Stratify by 'y' to ensure balanced representation of malicious samples in both sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples, Testing set: {len(X_test)} samples.")
print(f"Malicious samples in test set: {y_test.sum()}")

    # --- 4. Train Decision Tree Classifier Model ---
print("Training Decision Tree Classifier model...")
    # Decision Trees are interpretable and can capture non-linear relationships
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

    # --- 5. Make Predictions and Evaluate Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
    # The classification_report provides precision, recall, f1-score per class
print(classification_report(y_test, y_pred, target_names=['Normal', 'Malicious']))

print("\nConfusion Matrix:")
    # Confusion Matrix:
    # [[True Negatives, False Positives]
    #  [False Negatives, True Positives]]
print(confusion_matrix(y_test, y_pred))

    # --- 6. Demonstrate a Few Sample Predictions ---
print("\n--- Sample Predictions ---")
sample_indices = [0, 5, 10, 15, 20] # Select a few from the test set
for i in sample_indices:
        sample_X = X_test.iloc[[i]]
        actual_y = y_test.iloc[i]
        predicted_y = model.predict(sample_X)[0]

        actual_label = "Malicious" if actual_y == 1 else "Normal"
        predicted_label = "Malicious" if predicted_y == 1 else "Normal"
        print(f"Sample {i}: Actual = {actual_label}, Predicted = {predicted_label}")

print("\nProject 5 (Simple Binary Classification) completed.")
    