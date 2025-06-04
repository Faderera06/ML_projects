ML Project: Feature Importance for Threat Indicators (Simulated Network Traffic Analysis)

Project Goal
To train a classification model on simulated network traffic data and identify which features are most important for distinguishing between normal and malicious activity, thereby enhancing model interpretability and providing actionable insights for threat detection.

Methodology

Data Generation and Features
* The project utilizes **synthetically generated data** to simulate network traffic, encompassing both normal and malicious patterns.
* Features chosen to represent network activity and potential indicators of compromise include:
    * `packet_rate`: Rate of packets transmitted.
    * `error_rate`: Frequency of network errors.
    * `bytes_transferred`: Volume of data transferred.
    * `connection_attempts`: Number of connection attempts.
    * `alert_frequency`: How often security alerts are triggered.
* Malicious traffic was simulated with characteristics typically associated with attacks (e.g., high error rates, many failed connection attempts, elevated alert frequencies), differentiating it from normal traffic.

Preprocessing
* All numerical features were **scaled using `StandardScaler`** to standardize their ranges, ensuring that all features contribute equally to the model's learning process and feature importance calculations.
* The dataset was then **split into training and testing sets** using `train_test_split`, with stratification to maintain the class balance.

 Model Selection and Training
* A **RandomForestClassifier** was selected for this project. Random Forests are ensemble learning methods that build multiple decision trees and merge their predictions. A key advantage of Random Forests is their ability to inherently provide **feature importances**, indicating how much each feature contributes to the model's overall predictive power.
* The model was trained on the preprocessed training data, using `class_weight='balanced'` to handle the likely imbalance between normal and malicious samples.

 Model Evaluation and Feature Importance Analysis
* The trained model's classification performance was evaluated using **Accuracy Score** and a detailed **Classification Report** (precision, recall, F1-score for both classes) on the unseen test set.
* Crucially, the `feature_importances_` attribute of the trained RandomForestClassifier was extracted. This attribute provides a numerical score for each feature, representing its relative importance in the model's decision-making process.
* The features were then ranked by their importance, and the **top features were displayed** in the terminal output.
* A **bar chart** was generated using `matplotlib` and `seaborn` to visually represent the feature importances, providing an intuitive understanding of which factors are most critical for detecting malicious network traffic.

Key Findings and Achieved Results
This project successfully demonstrated the power of Random Forest in not only classifying network traffic but also in identifying the most influential threat indicators. The analysis revealed which simulated network behaviors (e.g., error rate, alert frequency, connection attempts) were most critical for distinguishing malicious activity. This capability is vital for building transparent and actionable security intelligence systems, directly supporting the goal of understanding the drivers behind detected anomalies in complex environments like XR.

 Technologies Used
* Python
* `scikit-learn` (RandomForestClassifier, StandardScaler, train_test_split, classification_report, accuracy_score)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python feature_importance_analysis.py`
