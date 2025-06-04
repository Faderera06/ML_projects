ML Project: Time-Series Anomaly Detection (User Login Behavior)

Project Goal
To develop and evaluate an unsupervised anomaly detection model capable of identifying unusual login patterns in synthetically generated time-series data, providing a foundation for continuous behavioral monitoring.

Methodology

Data Generation and Features
* The project utilizes synthetically generated time-series data** to simulate user login events over a period.
* "Normal" login events were generated to predominantly occur within typical business hours (e.g., 9 AM - 5 PM) from Monday to Friday.
* "Anomalous" login events were strategically injected to occur outside these normal patterns (e.g., very early morning or late-night logins, or logins on weekends).
* Key features extracted from the timestamps for anomaly detection were `Hour` of the day and `DayOfWeek`.

Preprocessing
* The extracted numerical features (`Hour`, `DayOfWeek`) were **scaled using `StandardScaler`** to normalize their ranges, ensuring equal contribution to the anomaly detection process.

Model Selection and Training
* Isolation Forest was chosen as the unsupervised anomaly detection algorithm. Isolation Forest is highly effective for anomaly detection, particularly in high-dimensional datasets, as it works by isolating anomalies rather than profiling normal data. It builds decision trees that recursively partition the data, and anomalies are typically isolated in fewer splits, making them easier to identify.
* The Isolation Forest model was trained on the scaled login data.

Model Evaluation and Visualization
* As an unsupervised method, evaluation relies on the model's ability to assign an anomaly score. The `predict` method of Isolation Forest directly labels points as inliers (1) or outliers (-1).
* A scatter plot was generated using `matplotlib` and `seaborn`, visualizing the login events based on `Hour` and `DayOfWeek`. Detected anomalies were highlighted in a distinct color (red 'X' marks), providing a clear visual representation of temporal outliers.

Key Findings and Achieved Results
This project successfully demonstrated the application of Isolation Forest for time-series anomaly detection in user login behavior. The model effectively identified the synthetically injected anomalous login patterns, showcasing its capability to learn and flag deviations from normal temporal activity. This skill is directly transferable to analyzing sequential XR telemetry data for suspicious user behaviors over time.

Technologies Used
* Python
* `scikit-learn` (Isolation Forest, StandardScaler)
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

How to Run
1.  Ensure you have Python and the listed libraries installed (preferably in a virtual environment).
2.  Execute the Python script: `python login_anomaly_detection.py`
