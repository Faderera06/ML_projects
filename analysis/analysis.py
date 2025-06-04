import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Good for feature importance
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Generate Synthetic Network Traffic Data ---
print("Generating synthetic network traffic data...")
np.random.seed(42)

# Features: 'packet_rate', 'error_rate', 'bytes_transferred', 'connection_attempts', 'alert_frequency'
# Target: 'is_malicious' (0 for normal, 1 for malicious)

num_normal_samples = 900
num_malicious_samples = 100

# Normal data: typical, low-risk patterns
normal_data = pd.DataFrame({
    'packet_rate': np.random.normal(loc=1000, scale=150, size=num_normal_samples),
    'error_rate': np.random.normal(loc=0.01, scale=0.005, size=num_normal_samples).clip(0, 0.05),
    'bytes_transferred': np.random.normal(loc=50000, scale=10000, size=num_normal_samples),
    'connection_attempts': np.random.randint(5, 20, size=num_normal_samples),
    'alert_frequency': np.random.randint(0, 2, size=num_normal_samples) # Low alerts
})
normal_data['is_malicious'] = 0

# Malicious data: patterns indicative of threats (e.g., high error rate, many failed attempts, high alerts)
malicious_data = pd.DataFrame({
    'packet_rate': np.random.normal(loc=200, scale=50, size=num_malicious_samples), # Low packet rate (e.g., C2 traffic)
    'error_rate': np.random.normal(loc=0.5, scale=0.1, size=num_malicious_samples).clip(0.1, 1.0), # High errors
    'bytes_transferred': np.random.normal(loc=1000, scale=500, size=num_malicious_samples), # Small transfers
    'connection_attempts': np.random.randint(50, 150, size=num_malicious_samples), # Many failed attempts
    'alert_frequency': np.random.randint(5, 15, size=num_malicious_samples) # High alerts
})
malicious_data['is_malicious'] = 1

# Combine and shuffle data
df = pd.concat([normal_data, malicious_data], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Generated {len(df)} total network records ({df['is_malicious'].sum()} malicious).")

# Define features (X) and target (y)
X = df.drop('is_malicious', axis=1)
y = df['is_malicious']

# --- 2. Preprocessing: Scale Numerical Features ---
print("Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# --- 3. Split Data into Training and Testing Sets ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples, Testing set: {len(X_test)} samples.")
print(f"Malicious samples in test set: {y_test.sum()}")

# --- 4. Train RandomForestClassifier Model ---
print("Training RandomForestClassifier model...")
# RandomForestClassifier is an ensemble model that provides feature importances
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # 'balanced' for imbalanced data
model.fit(X_train, y_train)

# --- 5. Make Predictions and Evaluate Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Malicious']))

# --- 6. Extract and Visualize Feature Importances ---
print("\n--- Feature Importance Analysis ---")
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Top 5 Features by Importance:")
print(importance_df.head())

# Visualize feature importances
plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Malicious Traffic Detection')
plt.xlabel('Importance (Gini Importance)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_plot.png') # Save the plot
plt.show()

print("\nProject 6 (Feature Importance Analysis) completed.")
