ML Project 4: User Behavior Clustering (Simulated Activity Segmentation)

Project Goal
To implement and evaluate a clustering algorithm to segment simulated users into distinct behavioral groups based on their activity patterns, demonstrating the ability to uncover hidden structures in user data.

Methodology

Data Generation and Features
The project utilizes synthetically generated data to simulate various user activity profiles.
Three distinct groups of users were simulated, each characterized by different distributions across features such as:
    * `daily_activity_score`: Overall engagement level.
    * `avg_session_duration_min`: Average time spent in sessions.
    * `interactions_per_session`: Frequency of interactions within a session.
This setup allowed for a controlled environment to test the clustering algorithm's ability to recover these known underlying groups.

Preprocessing
* The generated features were **scaled using `StandardScaler`** to normalize their ranges. This is a critical step for distance-based clustering algorithms like K-Means, as it prevents features with larger numerical values from disproportionately influencing the clustering process.

Model Selection and Training
* K-Means Clustering** was chosen as the unsupervised learning algorithm. K-Means is a popular and efficient algorithm for partitioning `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (centroid).
* The model was configured to identify `k=3` clusters, aligning with the three distinct user groups simulated in the data generation phase. The `n_init=10` parameter was used to run the algorithm multiple times with different centroid seeds, selecting the best result for more robust cluster assignments.
* The K-Means model was trained on the scaled user activity data.

Model Evaluation and Visualization
* As an unsupervised method, evaluation is primarily based on the coherence of the resulting clusters and visual inspection.
* The cluster centers (means of features for each cluster) were printed, providing insights into the characteristics of each identified group.
* A scatter plot was generated using `matplotlib` and `seaborn`, visualizing the users based on two key features (`daily_activity_score` vs. `avg_session_duration_min`). Each user point was colored according to its assigned cluster, visually demonstrating the algorithm's success in segmenting the data into distinct groups.

Key Findings and Achieved Results
This project successfully demonstrated the application of K-Means clustering for user behavior segmentation. The algorithm effectively identified and separated the three distinct user profiles present in the synthetic data. This capability is invaluable for understanding diverse user populations in XR environments, allowing for the establishment of tailored "normal" baselines for different user types, which can then enhance the precision of anomaly detection systems.

Technologies Used
* Python
* `scikit-learn` (K-Means, StandardScaler)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python user_clustering.py`
