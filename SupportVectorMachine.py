import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve
import matplotlib.pyplot as plt

# Load the data from the Excel file
df = pd.read_excel('data-4-6.xlsx')

# Extract the features and labels from the data
X = df[['City', 'Highway']].values
y = df['Engine'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM model
clf = SVC(kernel='linear', C=1)

# Train the model and time it
alg3StartTime = time.time()
clf.fit(X_train, y_train)
alg3EndTime = time.time()

alg3TrainTime = alg3EndTime - alg3StartTime

# Make predictions on the testing data
alg3StartTime = time.time()
y_pred = clf.predict(X_test)
alg3EndTime = time.time()

alg3TestTime = alg3EndTime - alg3StartTime

# compute the error rate
accuracy = accuracy_score(y_test, y_pred) * 100  # Calculate the accuracy of the predictions
error_rate = 100 - accuracy  # Calculate the error rate (the complement of the accuracy)


# Compute the computational complexity
n_samples = X.shape[0]
n_features = X.shape[1]
n_classes = len(np.unique(y))

complexity = (n_samples * n_features + n_features ** 2) * n_classes + n_classes ** 2



#prints
print('Support Vector Machine Training time:\t', alg3TrainTime, "seconds")
print('Support Vector Machine Testing time:\t', alg3TestTime, "seconds")
print('Computational Complexity:\t\t\t\t', complexity)
print("Correct Predictions:\t\t\t\t\t",(y_pred == y_test).sum(), "/", len(y_test))
print('Accuracy:\t\t\t\t\t\t\t\t', accuracy, "%")
print("Error rate:\t\t\t\t\t\t\t\t", error_rate, "%")


confMatrixAlg3 = confusion_matrix(y_test,y_pred)

# Compute the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

