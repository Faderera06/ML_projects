ML Project 5: Simple Binary Classification (Simulated Security Alert Classification)


Project Goal
To develop and rigorously evaluate a supervised machine learning model capable of accurately classifying simulated security alerts as either normal or malicious, based on a set of predefined features.

Methodology

Data Generation and Features
The project utilizes synthetically generated data to simulate security alerts, comprising both "normal" and "malicious" instances.
Features were designed to represent typical alert characteristics:
    * `packet_size`: Size of network packets.
    * `connection_duration`: Duration of network connections.
    * `failed_attempts`: Number of failed login or access attempts.
    * `source_reputation`: A score indicating the trustworthiness of the source.
* Malicious samples were generated with patterns typically associated with attacks (e.g., small packet sizes, short connection durations, many failed attempts, low source reputation), while normal samples reflected benign activity. The dataset was intentionally imbalanced to reflect real-world security scenarios (fewer malicious samples).

Preprocessing
* All numerical features were **scaled using `StandardScaler`** to normalize their ranges, ensuring that no single feature disproportionately influenced the model's learning process.
* The dataset was then **split into training and testing sets** using `train_test_split`, with `stratify=y` to ensure that both sets maintained a representative proportion of the minority (malicious) class.

Model Selection and Training
* A **Decision Tree Classifier** was chosen for this project. Decision Trees are interpretable models that can capture non-linear relationships in data. They are relatively simple to understand, making them suitable for initial classification tasks where insights into decision rules are valuable.
* The Decision Tree model was trained on the preprocessed training data.

Model Evaluation
The trained model's performance was rigorously evaluated on the unseen test set using standard classification metrics:
* Accuracy Score: The overall proportion of correct predictions.
* Classification Report: Provides detailed metrics including Precision, Recall, and F1-score for both the "Normal" and "Malicious" classes. Precision indicates the proportion of positive identifications that were actually correct, while Recall indicates the proportion of actual positives that were correctly identified. The F1-score is the harmonic mean of Precision and Recall. For security, Recall for the "Malicious" class is often critical (minimizing false negatives).
* Confusion Matrix: A table that visualizes the performance of the classification model, showing counts of True Positives, True Negatives, False Positives, and False Negatives.
* Sample Predictions: A few individual test samples were predicted to demonstrate the model's classification ability firsthand.

Key Findings and Achieved Results
This project successfully demonstrated the application of binary classification for simulated security alert triage. The Decision Tree model exhibited its capability to learn patterns and distinguish between normal and malicious events, even with an imbalanced dataset. The detailed evaluation metrics provided clear insights into the model's strengths and weaknesses, highlighting its effectiveness in identifying potential threats. This foundational classification skill is directly applicable to categorizing various types of anomalies and threats detected in XR environments.

Technologies Used
* Python
* `scikit-learn` (DecisionTreeClassifier, StandardScaler, train_test_split, classification_report, accuracy_score, confusion_matrix)
* `pandas`
* `numpy`

How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python security_alert_classifier.py`
