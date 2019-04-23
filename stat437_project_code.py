import pandas_montecarlo
import numpy as np
from scipy.stats import kruskal
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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
#Scale all features to be with 0 and 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification with different algorithms

print("actual data")
#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SV:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
from sklearn.svm import SVC
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SV Kerne:", svk.score(X_test,Y_test))

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 1234)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_cm = confusion_matrix(Y_test, DT_Y_pred)
print(DT_cm)
print("Accuracy for Decission Tree:", DT.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))
#################################################################################################

#Step 2B
#Use data filtered based on PCA and important features for classification

#Using pca_data
print("PCA data")
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
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SV:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
from sklearn.svm import SVC
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SV Kerne:", svk.score(X_test,Y_test))

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 1234)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_cm = confusion_matrix(Y_test, DT_Y_pred)
print(DT_cm)
print("Accuracy for Decission Tree:", DT.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))
########################################################################################################################################

#Using random forest selected features data
print("rndf data")

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
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SV:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
from sklearn.svm import SVC
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SV Kerne:", svk.score(X_test,Y_test))

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 1234)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_cm = confusion_matrix(Y_test, DT_Y_pred)
print(DT_cm)
print("Accuracy for Decission Tree:", DT.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))
################################################################################################################

#Using Recursive Feature Elimination data
print("RFE data")
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
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SV:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
from sklearn.svm import SVC
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SV Kerne:", svk.score(X_test,Y_test))

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 1234)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_cm = confusion_matrix(Y_test, DT_Y_pred)
print(DT_cm)
print("Accuracy for Decission Tree:", DT.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))
##################################################################################################

#Using Correlational filtered data
print("correlational filtered data")

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
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 1234, solver = "liblinear")
LR.fit(X_train, Y_train)
LR_Y_pred = LR.predict(X_test)
LR_cm = confusion_matrix(Y_test, LR_Y_pred)
print("Confusion Matrix")
print(LR_cm)
print("classification Accuracy for Logistic Regression:", LR.score(X_test,Y_test))


#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
knn_Y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(Y_test, knn_Y_pred)
print(knn_cm)
print("Classification Accuracy for KNN:", knn.score(X_test,Y_test))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
sv = SVC(kernel = 'linear', random_state = 1234)
sv.fit(X_train, Y_train)
sv_Y_pred = sv.predict(X_test)
sv_cm = confusion_matrix(Y_test, sv_Y_pred)
print(sv_cm)
print("Accuracy for SV:", sv.score(X_test,Y_test))

#Using SVC method of svm class to use Kernel method of SVM Algorithm
from sklearn.svm import SVC
svk = SVC(kernel = 'rbf', random_state = 1234)
svk.fit(X_train, Y_train)
svk_Y_pred = svk.predict(X_test)
svk_cm = confusion_matrix(Y_test, svk_Y_pred)
print(svk_cm)
print("Accuracy for SV Kerne:", svk.score(X_test,Y_test))

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 1234)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_cm = confusion_matrix(Y_test, DT_Y_pred)
print(DT_cm)
print("Accuracy for Decission Tree:", DT.score(X_test,Y_test))

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_cm = confusion_matrix(Y_test, RF_Y_pred)
print(RF_cm)
print("Accuracy for Random Forest:", RF.score(X_test,Y_test))

#Because of how well these models performed at classifying the data
#We choose to proceed all
#Logistic Regression, KNN, SVM Kernel, and Random Forest

#########################################################################################################################################
#Use the monte carlo simulation to simulate feature values
#import the rndf data created to be used for monte carlo simulation
mc_data_M = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\mc_data_M.csv")
mc_data_B = pd.read_csv(r"C:\Users\okoro\OneDrive\Desktop\STAT 437\mc_data_B.csv")

#columns = ["area_worst", "concave.points_mean", "concave.points_worst", "perimeter_worst", "radius_worst"]

#Run monte carlo simulations
#for Malignant (M)
mc_aw_M = mc_data_M["area_worst"].montecarlo(sims=20)
mc_cm_M = mc_data_M["concave.points_mean"].montecarlo(sims=20)
mc_cw_M = mc_data_M["concave.points_worst"].montecarlo(sims=20)
mc_pw_M = mc_data_M["perimeter_worst"].montecarlo(sims=20)
mc_rw_M = mc_data_M["radius_worst"].montecarlo(sims=20)

#Use kruskal to check that data has the same distribution with original
kruskal(mc_aw_M.data.iloc[:,0], mc_aw_M.data.iloc[:,-1])

#for Benign (B)
mc_aw_B = mc_data_B["area_worst"].montecarlo(sims=20)
mc_cm_B = mc_data_B["concave.points_mean"].montecarlo(sims=20)
mc_cw_B = mc_data_B["concave.points_worst"].montecarlo(sims=20)
mc_pw_B = mc_data_B["perimeter_worst"].montecarlo(sims=20)
mc_rw_B = mc_data_B["radius_worst"].montecarlo(sims=20)

#Use kruskal to check that data has the same distribution with original
kruskal(mc_aw_B.data.iloc[:,0], mc_aw_M.data.iloc[:,-1])



#mc_df = pd.DataFrame()


#mc = data["radius_mean"].montecarlo(sims=20)
#data.replace({"diagnosis": "M"}, 1, inplace=True)
#data.replace({"diagnosis": "B"}, 0, inplace=True)
"""
y = data["diagnosis"]

data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1, inplace=True)
feature_list = list(data.columns)

#Variable Importance

rf_exp = RandomForestRegressor(n_estimators= 1000, random_state=100)
rf_exp.fit(data, y)
importances = list(rf_exp.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]


x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable');
plt.title('Variable Importances');

plt.show()
"""
