Project: Network Intrusion Detection System

Project Goal
To develop and evaluate binary classification models capable of accurately distinguishing between legitimate and intrusive network connections using a well-known cybersecurity dataset.


 Data Acquisition and Preprocessing
The dataset used for this project is the NSL-KDD dataset, a refined version of the KDD Cup 1999 dataset, specifically designed for network intrusion detection. It can be accessed and downloaded from Kaggle: [https://www.kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd).
* The dataset contains various features describing network connections (e.g., duration, protocol type, service, flag, number of failed logins).
* Categorical features were identified and one-hot encoded to convert them into a numerical format suitable for machine learning algorithms.
* Numerical features were scaled using `StandardScaler` to normalize their ranges, preventing features with larger values from dominating the model training.
* The dataset was split into training and testing sets to ensure robust model evaluation.

Model Selection and Training
Two common binary classification algorithms were selected for their interpretability and effectiveness in various cybersecurity applications:
* Logistic Regression: A linear model used for binary classification, providing probabilities of an event occurring.
* Decision Tree Classifier: A non-linear model that makes decisions based on a tree-like structure of rules, offering high interpretability.
Both models were trained on the preprocessed training data.

Model Evaluation
The trained models were evaluated on the unseen test set using standard classification metrics:
* Accuracy Score: Overall correctness of the model.
* Classification Report:** Provides precision, recall, and F1-score for each class (normal and intrusion), offering a detailed view of the model's performance, especially crucial for detecting the minority class (intrusions).
* Confusion Matrix: Visualizes the counts of true positive, true negative, false positive, and false negative predictions.

 Key Findings and Achieved Results
The project successfully demonstrated the application of machine learning for network intrusion detection. Both Logistic Regression and Decision Tree models were able to identify intrusive network activities with reasonable accuracy. The classification reports provided insights into the models' ability to correctly identify true intrusions (recall) and avoid false alarms (precision). For instance, the Decision Tree often showed strong performance due to its ability to capture complex, non-linear relationships in the network features. This project showcases proficiency in data preprocessing, model selection, training, and evaluation for cybersecurity applications.

 Technologies Used
* Python
* `scikit-learn` (Logistic Regression, Decision Tree Classifier, StandardScaler, train_test_split, classification_report, confusion_matrix)
* `pandas`
* `numpy`

 How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Download the NSL-KDD dataset and place it in the same directory as the script, or modify the script to point to its location.
3.  Execute the Python script: `python network_intrusion_detection.py`
