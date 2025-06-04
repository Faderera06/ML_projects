ML Multivariate Anomaly Detection (Simulated XR Telemetry)

Project Goal
To develop and evaluate an unsupervised anomaly detection model capable of identifying anomalous events in simulated multi-dimensional XR telemetry data, providing a foundation for comprehensive spatial cyber threat monitoring.
Methodology

Data Generation and Features
The project utilizes **synthetically generated data** to simulate both normal and anomalous XR telemetry streams, capturing the multivariate nature of real XR environments.
Features selected to represent typical XR user activity include:
    * `head_rot_speed`: Speed of head rotation (e.g., in radians per second).
    * `hand_dist_moved`: Distance hands have moved (e.g., in meters per second).
    * `gaze_dwell_time_ms`: How long the user's gaze rests on a point (in milliseconds).
    * `interaction_rate`: Frequency of user interactions (e.g., button presses, object grabs).
Normal data was simulated to represent coherent, typical user behavior. Anomalies were strategically injected as unusual combinations of these features (e.g., very high hand movement with very low head rotation, or abnormally high interaction rates with very short gaze dwell times), mimicking potential attacks or system glitches.

Preprocessing
* All numerical features were **scaled using `StandardScaler` to normalize their ranges. This is crucial for distance-based algorithms like OCSVM, as it prevents features with larger numerical values from disproportionately influencing the decision boundary.

Model Selection and Training
* One-Class Support Vector Machine (OCSVM) was chosen as the unsupervised anomaly detection algorithm. OCSVM is particularly suited for multivariate anomaly detection where only the 'normal' class is well-defined. It learns a decision boundary that encapsulates the majority of the normal data points, effectively classifying any data point falling outside this learned boundary as an anomaly.
* The `kernel='rbf'` (Radial Basis Function) was used for its ability to model non-linear decision boundaries. The `nu` parameter was set to 0.03, implying an expectation that approximately 3% of the data points are outliers, guiding the model's sensitivity.
* The OCSVM model was trained on the scaled XR telemetry data, learning the intricate normal patterns.

Model Evaluation and Visualization
* As OCSVM is an unsupervised method, evaluation is primarily based on the count of detected anomalies and visual inspection of their location within the feature space. The `predict` method assigns labels (-1 for outlier, 1 for inlier).
* A scatter plot was generated using `matplotlib` and `seaborn`, visualizing two key features from the simulated telemetry (e.g., `head_rot_speed` vs. `hand_dist_moved`). Detected anomalies were highlighted in a distinct color (red 'X' marks), clearly demonstrating the OCSVM's ability to identify multivariate outliers that deviate from the learned normal behavior distribution.

Key Findings and Achieved Results
This project successfully demonstrated the application of One-Class SVM for multivariate anomaly detection in simulated XR telemetry. The model effectively identified complex behavioral anomalies that might not be apparent in individual feature streams, showcasing its potential for detecting subtle, interconnected deviations indicative of security threats or system issues in immersive environments. This technique is a crucial component for building a robust, continuous spatial cyber threat monitoring system for XR.

Technologies Used
* Python
* `scikit-learn` (OneClassSVM, StandardScaler)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python multivariate_anomaly_detection.py`
