
# K-Nearest Neighbors (KNN) Classifier Project

This project demonstrates the implementation of a K-Nearest Neighbors (KNN) classifier using artificially generated data. The key steps include data exploration, data preprocessing, model training, evaluation, and hyperparameter tuning.


## Table of Contents

This dataset is taken from Kaggle. The dataset includes the following columns:

    1. Installation
    2. Data Exploration
    3. Data Preprocessing
    4. Model Training
    5. Model Evaluation
    6. Hyperparameter Tuning
    7. Retraining with Optimal K Value
    8. Conclusion.


# Project Workflow
## Installation

Install my-project with npm

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline

```
    
## Data Exploration

Since the data is artificial, we start with exploratory data analysis (EDA). A pair plot is generated to visualize the data:
```bash
sns.pairplot(df, hue='TARGET CLASS')
plt.show()

```

## Data Preprocessing

The features are standardized to have zero mean and unit variance:
```bash
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()



```
## Model Training

The data is split into training and testing sets:

``` bash
 X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)
```

A KNN model is instantiated and trained with n_neighbors=1:
``` bash
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
```

## Model Evaluation

Predictions are made on the test set and evaluated using a confusion matrix and classification report:

``` bash
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```
### Output:

```Confusion Matrix:
[[112  40]
 [ 34 114]]

Classification Report:
             precision    recall  f1-score   support
          0       0.77      0.74      0.75       152
          1       0.74      0.77      0.75       148
avg / total       0.75      0.75      0.75       300
```
##3 Hyperparameter Tuning

The elbow method is used to determine the optimal K value by plotting the error rate for different K values:

```
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```
### Retraining with Optimal K Value

Based on the elbow plot, the model is retrained with n_neighbors=30 and evaluated again:
```
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```
### Output:

```Confusion Matrix:
[[127  25]
 [ 23 125]]

Classification Report:
             precision    recall  f1-score   support
          0       0.85      0.84      0.84       152
          1       0.83      0.84      0.84       148
avg / total       0.84      0.84      0.84       300
```
# Conclusion

The K-Nearest Neighbors algorithm was implemented, evaluated, and optimized on artificially generated data. By tuning the K value, the model's performance was improved, demonstrating the importance of hyperparameter tuning in machine learning.