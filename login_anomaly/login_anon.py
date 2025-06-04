import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

    # --- 1. Generate Synthetic Time-Series Login Data ---
print("Generating synthetic login data...")
np.random.seed(42)

    # Generate normal login times (mostly within business hours, some scattered)
num_normal_days = 30
normal_logins_per_day = 50
normal_data = []

for day in range(num_normal_days):
        base_time = pd.to_datetime(f'2024-01-{day+1:02d} 09:00:00')
        # Simulate logins mostly between 9 AM and 5 PM
login_hours = np.random.normal(loc=13, scale=2, size=normal_logins_per_day)
login_hours = np.clip(login_hours, 9, 17) # Clamp to business hours
        
for hour in login_hours:
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            timestamp = base_time + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
            normal_data.append({'Timestamp': timestamp})

df_normal = pd.DataFrame(normal_data)

    # Introduce some anomalies: logins outside business hours, very early/late
num_anomalies = 10
anomaly_data = []
for i in range(num_anomalies):
    day = np.random.randint(1, num_normal_days + 1)
        # Simulate very late or very early logins
hour = np.random.choice(np.concatenate([np.random.normal(loc=23, scale=1, size=5),
                                                np.random.normal(loc=2, scale=1, size=5)]))
hour = np.clip(hour, 0, 23) # Clamp to 24 hours
minute = np.random.randint(0, 60)
second = np.random.randint(0, 60)
timestamp = pd.to_datetime(f'2024-01-{day:02d} 00:00:00') + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
anomaly_data.append({'Timestamp': timestamp})

df_anomaly = pd.DataFrame(anomaly_data)

    # Combine and shuffle
df = pd.concat([df_normal, df_anomaly], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Generated {len(df)} total login records.")

    # Convert Timestamp to numerical features: Hour of Day, Day of Week
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek # Monday=0, Sunday=6

    # Select features for anomaly detection
features = ['Hour', 'DayOfWeek']
X = df[features]

    # --- 2. Preprocessing: Scale Features ---
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

    # --- 3. Train Isolation Forest Model ---
    # Isolation Forest is effective for anomaly detection.
    # It "isolates" anomalies by randomly picking a feature and then randomly picking a split value.
    # Anomalies are fewer and thus isolated closer to the root of the tree.
print("Training Isolation Forest model...")
model = IsolationForest(
        contamination='auto', # 'auto' lets the model estimate the proportion of outliers
        random_state=42,
        n_estimators=100 # Number of trees in the forest
    )
model.fit(X_scaled)

    # Predict anomalies (-1 for outlier, 1 for inlier)
df['anomaly'] = model.predict(X_scaled)
df['is_anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0) # Convert to 0/1

print(f"Detected {df['is_anomaly'].sum()} anomalies.")

    # --- 4. Visualize Results ---
print("Generating visualization...")
plt.figure(figsize=(12, 7))
sns.scatterplot(
        x=df['Hour'],
        y=df['DayOfWeek'],
        hue=df['is_anomaly'],
        palette={0: 'blue', 1: 'red'},
        s=100,
        alpha=0.7,
        style=df['is_anomaly'],
        markers={0: 'o', 1: 'X'}
    )
plt.title('Login Anomaly Detection using Isolation Forest')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week (0=Mon, 6=Sun)')
plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'])
plt.tight_layout()
plt.show()

print("\nProject 3 (Time-Series Anomaly Detection) completed.")