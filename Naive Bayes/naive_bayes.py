# Naive Bayes

# Importing the libraries
import pandas as pd
import time

# Importing the dataset
datatest = pd.read_csv('Test++.csv')
datatrain = pd.read_csv('Train++.csv')
X_train = datatrain.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 23, 24, 32, 33]].values
X_test = datatest.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 23, 24, 32, 33]].values
y_train = datatrain.iloc[:, 41].values
y_test = datatest.iloc[:, 41].values
y_test_df = pd.DataFrame(y_test)
y_train_df = pd.DataFrame(y_train)
y_test_df.to_csv("Test_Label.csv", encoding='utf-8', index=False)
y_train_df.to_csv("Train_Label.csv", encoding='utf-8', index=False)

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X_train[:, 0] = labelencoder_x.fit_transform(X_train[:, 0])
X_train[:, 1] = labelencoder_x.fit_transform(X_train[:, 1])
X_train_en = pd.DataFrame(X_train)

X_test[:, 0] = labelencoder_x.fit_transform(X_test[:, 0])
X_test[:, 1] = labelencoder_x.fit_transform(X_test[:, 1])
X_test_en = pd.DataFrame(X_test)
X_test_en.to_csv("Test_Encoder.csv", encoding='utf-8', index=False)
X_train_en.to_csv("Train_Encoder.csv", encoding='utf-8', index=False)

# Normalization
from sklearn.preprocessing import Normalizer
norm = Normalizer()
X_train = norm.fit_transform(X_train)
X_test = norm.fit_transform(X_test)
X_train_norm = pd.DataFrame(X_train)
X_test_norm = pd.DataFrame(X_test)

a = pd.DataFrame(y_train, columns=['Label'])
b = pd.DataFrame(y_test, columns=['Label'])
frame1 = [X_train_norm,a]
frame2 = [X_test_norm,b]
result_train = pd.concat(frame1, axis=1)
result_train.to_csv("Train_Norm.csv", encoding='utf-8', index=False)
resut_test = pd.concat(frame2, axis=1)
resut_test.to_csv("Test_Norm.csv", encoding='utf-8', index=False)

# Start Time
start = time.time()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Build Time
build = time.time()-start

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)

# Print
print("---------- Naive Bayes ----------")
print("True Possitive = " +str(cm[1][1]))
print("True Negative = " +str(cm[0][0]))
print("False Possitive = " +str(cm[0][1]))
print("False Negative = " +str(cm[1][0]))
print("Accuracy = " +str(acc))
print("Build Time = ",build)
print("\n")