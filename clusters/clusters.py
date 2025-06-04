import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

    # Generate Synthetic User Activity Data 
print("Generating synthetic user activity data...")
np.random.seed(42)

    # Simulate different user types:
    # Group 1: High activity, medium duration, frequent interactions (e.g., active user)
    # Group 2: Low activity, short duration, infrequent interactions (e.g., casual user)
    # Group 3: Medium activity, long duration, medium interactions (e.g., explorer user)

num_users_per_group = 100
total_users = num_users_per_group * 3

    # Features: 'daily_activity_score', 'avg_session_duration_min', 'interactions_per_session'
data = np.vstack([
        np.random.normal(loc=[80, 45, 15], scale=[10, 10, 5], size=(num_users_per_group, 3)), # Group 1
        np.random.normal(loc=[30, 15, 5], scale=[8, 5, 2], size=(num_users_per_group, 3)),   # Group 2
        np.random.normal(loc=[60, 70, 10], scale=[12, 15, 4], size=(num_users_per_group, 3)) # Group 3
    ])

    # Ensure all values are non-negative
data = np.clip(data, 0, None)

df = pd.DataFrame(data, columns=['daily_activity_score', 'avg_session_duration_min', 'interactions_per_session'])
print(f"Generated {len(df)} user records.")

    # Preprocessing: Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

    # Train K-Means Clustering Model 
  
print("Training K-Means clustering model (k=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
df['cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster centers (scaled features):")
print(kmeans.cluster_centers_)
print(f"\nAssigned {df['cluster'].nunique()} clusters.")
print("\nUsers per cluster:")
print(df['cluster'].value_counts().sort_index())


    # Visualize Results (using two features for 2D plot) 
print("Generating visualization...")
plt.figure(figsize=(10, 7))
sns.scatterplot(
        x='daily_activity_score',
        y='avg_session_duration_min',
        hue='cluster',
        palette='viridis', # A nice color palette
        data=df,
        s=100,
        alpha=0.8
    )
plt.title('User Behavior Clustering (K-Means)')
plt.xlabel('Daily Activity Score (Scaled)')
plt.ylabel('Average Session Duration (min) (Scaled)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Cluster')

plt.tight_layout()
plt.show()

    # Save the plot programmatically (optional, but good practice)
plt.savefig('user_clustering_plot.png')

print("\nProject 4 (User Behavior Clustering) completed.")
    
