# SCT_DS_3
Decision Tree Classifier to predict bank product purchase using Bank Marketing dataset 
# SCT_DS_3 - Decision Tree Classifier

## Task
Build a decision tree classifier to predict whether a customer
will purchase a product based on demographic and behavioral data.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Bank Marketing (bank-full.csv)
- Size: 45,211 rows, 17 columns

## Steps
- Loaded and explored the dataset
- Encoded all categorical variables using LabelEncoder
- Split data into 80% train / 20% test
- Trained DecisionTreeClassifier with max_depth=5
- Evaluated with Accuracy, Precision, Recall, F1 Score

## Results
- Accuracy:  89.35%
- Precision: 59%
- Recall:    40%
- F1 Score:  48%
- Most important feature: call duration

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib
