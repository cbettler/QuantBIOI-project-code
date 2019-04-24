import pandas as pd
import numpy
import pandas_montecarlo
from scipy.stats import shapiro, kruskal, f_oneway
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression as lgr
import numpy as np

## RandomForest Classifier with monte carlo simulated training set
np.random.seed(1234)

df = pd.read_csv("data.csv")
df = df.drop(["id","Unnamed: 32"],axis=1)
df = df.replace({'diagnosis': "M"}, 1)
df = df.replace({'diagnosis': "B"}, 0)

#split dataset for mc seed and testing

df_mc, df = numpy.split(df, [int(.7*len(df))])

#split dataset by class
df_1 = df_mc.loc[df_mc.diagnosis==1]
df_0 = df_mc.loc[df_mc.diagnosis==0]
df_1 = df_1.drop(["diagnosis"],axis=1)
df_0 = df_0.drop(["diagnosis"],axis=1)

#simulate class 0 data
mc_sim_df_0 = pd.DataFrame()
mc_sim_df_0['diagnosis']= ['0'] * len(df_0.index)
for col in df_0.columns:
    col_sim = df_0[col].montecarlo(sims = 2, bust = 0, goal = 0).data
    col_sim = col_sim.drop(["original"],axis = 1)
    for col2 in col_sim.columns:
        mc_sim_df_0[col]=col_sim[col2]
#if(shapiro(mc_sim_df_1[col])[1]>0.05):
#print(kruskal(mc_sim_df_1[col],df_1[col]))
#else:
#print(f_oneway(mc_sim_df_1[col],df_1[col]))

#simulate class 1 data
mc_sim_df_1 = pd.DataFrame()
mc_sim_df_1['diagnosis']= ['1'] * len(df_1.index)
for col in df_1.columns:
    col_sim = df_1[col].montecarlo(sims = 2, bust = 0, goal = 0).data
    col_sim = col_sim.drop(["original"],axis = 1)
    for col2 in col_sim.columns:
        mc_sim_df_1[col]=col_sim[col2]
#if(shapiro(mc_sim_df_1[col])[1]>0.05):
#print(kruskal(mc_sim_df_1[col],df_1[col]))
#else:
#print(f_oneway(mc_sim_df_1[col],df_1[col]))


#diag = mc_sim_df_1.append(mc_sim_df_0)['diagnosis']
mc_sim_df = mc_sim_df_1.append(mc_sim_df_0)
#shuffling dataframe for good luck
#mc_sim_df = mc_sim_df.sample(frac=1)
#mc_sim_df['diagnosis']=diag
mc_sim_df.head(20)


#values formatted
labels = df["diagnosis"]
df = df.drop("diagnosis",axis=1)
dfDev, dfTes = numpy.split(df, [int(.7*len(df))])
DDev, DTes = numpy.split(labels, [int(.7*len(labels))])

DTrn =  mc_sim_df['diagnosis']
dfTrn = mc_sim_df.drop(['diagnosis'], axis = 1)

#run model and test
#randomforest
model = rfc()
model = model.fit(dfTrn.values,DTrn)
pd = model.predict(dfDev)
hit = 0
for i in range(len(pd)):
    if(int(pd[i])==int(DDev.iloc[i])):
        hit+=1
print("random forest", hit/len(pd))

#knn
model = knc()
model = model.fit(dfTrn.values,DTrn)
pd = model.predict(dfDev)
hit = 0
for i in range(len(pd)):
    if(int(pd[i])==int(DDev.iloc[i])):
        hit+=1
print("knn", hit/len(pd))

#svc
model = svc()
model = model.fit(dfTrn.values,DTrn)
pd = model.predict(dfDev)
hit = 0
for i in range(len(pd)):
    if(int(pd[i])==int(DDev.iloc[i])):
        hit+=1
print("svc", hit/len(pd))

#logistic regression
model = lgr()
model = model.fit(dfTrn.values,DTrn)
pd = model.predict(dfDev)
hit = 0
for i in range(len(pd)):
    if(int(pd[i])==int(DDev.iloc[i])):
        hit+=1
print("logistic regression", hit/len(pd))
