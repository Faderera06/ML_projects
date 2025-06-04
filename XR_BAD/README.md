ML Project : Simulated XR User Behavior Anomaly Detection

Project Goal
To implement an unsupervised machine learning model that can effectively detect anomalous patterns in synthetic XR user telemetry data, providing a foundation for continuous spatial cyber threat monitoring.

Methodology

Data Generation and Features
Since real-world, labeled XR threat datasets are scarce, this project utilizes **synthetically generated data** to simulate typical and anomalous XR user behaviors. The simulated telemetry includes:
* `x_coordinate`, `y_coordinate`: Representing the user's spatial position.
* `interaction_frequency`: How often a user interacts with virtual objects.
* `gaze_deviation`: A measure of how erratic or unusual the user's gaze patterns are.
Normal behavior was simulated to cluster within expected ranges, while anomalies were generated to fall outside these clusters, representing suspicious movements or interactions.

 Preprocessing
* The generated features were scaled using `StandardScaler` to ensure that all features contributed equally to the anomaly detection process, preventing features with larger numerical ranges from dominating the distance calculations.
 Model Selection and Training
* Local Outlier Factor (LOF) was chosen as the unsupervised anomaly detection algorithm. LOF is effective at identifying anomalies by measuring the local deviation of a data point with respect to its neighbors. It works well in scenarios where anomalies are scattered or where the density of normal data varies.
* The LOF model was trained on the scaled synthetic XR behavior data.

 Model Evaluation and Visualization
* As LOF is unsupervised, evaluation is primarily visual and based on the model's ability to assign outlier scores. The `negative_outlier_factor_` attribute was used to identify anomalies.
* A **scatter plot** was generated to visualize the data in a 2D feature space (e.g., `x_coordinate` vs. `y_coordinate`), with detected anomalies highlighted in a distinct color. This visual representation clearly demonstrates the model's effectiveness in identifying outliers within the simulated XR environment.

 Key Findings and Achieved Results
This project successfully demonstrated the application of unsupervised anomaly detection for identifying unusual XR user behaviors. The LOF model effectively flagged the synthetically injected anomalies, showcasing its potential to detect subtle deviations in spatial and interaction patterns within immersive environments. This provides a crucial step towards building proactive, continuous threat monitoring systems for XR, where traditional signature-based detection is often insufficient.

 Technologies Used
* Python
* `scikit-learn` (Local Outlier Factor, StandardScaler)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

 How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python xr_behavior_anomaly_detection.py`
