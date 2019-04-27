import pandas_montecarlo
import numpy as np
from scipy.stats import kruskal
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Step 1
#import all data from R
data = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\cancer_data.csv")
pca_data = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\pc_data.csv")
rndf_filt_data = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\rndf_filt_data.csv")
rfe_filt_data = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\rfe_filt_data.csv")
cor_filt_data = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\cor_filt_data.csv")

#Step 2A
#Actual Cancer dataset
X = data.iloc[:,2:].values
Y = data.iloc[:,1].values

#Recode the M and B as M =1, B = 0
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split data into train 80% and test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

len(X_train)
len(X_test)

#Feature Scaling
#Scale all features to be within 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

print("actual original data with all dimensions retained")
#Using Logistic Regression Algorithm to the Training Set
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SVC:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SVM Kernel:", svk.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))
print("\n")

#################################################################################################

#Step 2B
#Use data filtered based on PCA and important features for classification

#Using pca_data
print("PCA Filtered data")
X = pca_data.iloc[:,2:].values
Y = pca_data.iloc[:,1].values

#Recode the M and B as M =1, B = 0
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split data into train 80% and test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

len(X_train)
len(X_test)

#Feature Scaling
#Scale all features to be with 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

#Using Logistic Regression Algorithm to the Training Set
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SVC:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SVM Kernel:", svk.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))

print("\n")
########################################################################################################################################

#Using random forest selected features data
print("Random Forest (rndf) filtered data")

X = rfe_filt_data.iloc[:,2:].values
Y = rfe_filt_data.iloc[:,1].values

#Recode the M and B as M =1, B = 0
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split data into train 80% and test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

len(X_train)
len(X_test)

#Feature Scaling
#Scale all features to be with 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

#Using Logistic Regression Algorithm to the Training Set
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SVC:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SVM Kernel:", svk.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))

print("\n")
################################################################################################################

#Using Recursive Feature Elimination data
print("RFE filtered data")
X = pca_data.iloc[:,2:].values
Y = pca_data.iloc[:,1].values

#Recode the M and B as M =1, B = 0
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split data into train 80% and test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

len(X_train)
len(X_test)

#Feature Scaling
#Scale all features to be with 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

#Using Logistic Regression Algorithm to the Training Set
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SVC:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SVM Kernel:", svk.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))

print("\n")
##################################################################################################

#Using Correlational filtered data
print("correlational matrix filtered data")

X = cor_filt_data.iloc[:,2:].values
Y = cor_filt_data.iloc[:,1].values

#Recode the M and B as M =1, B = 0
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split data into train 80% and test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

len(X_train)
len(X_test)

#Feature Scaling
#Scale all features to be with 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

#Using Logistic Regression Algorithm to the Training Set
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SVC:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SVM Kernel:", svk.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))

#Because of how well these models performed at classifying the data
#We choose to proceed with all
#Logistic Regression, KNN, SVM, and Random Forest
