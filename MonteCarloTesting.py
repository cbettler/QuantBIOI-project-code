import pandas as pd
import numpy
import pandas_montecarlo
from scipy.stats import shapiro, kruskal, f_oneway
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression as lgr

def classify(s):
    import pandas as pd
    import numpy
    import pandas_montecarlo
    from scipy.stats import shapiro, kruskal, f_oneway
    from sklearn.ensemble import RandomForestClassifier as rfc
    from sklearn.neighbors import KNeighborsClassifier as knc
    from sklearn.svm import SVC as svc
    from sklearn.linear_model import LogisticRegression as lgr
    ## RandomForest Classifier with monte carlo simulated training set
    numpy.random.seed(s)

    #df = pd.read_csv("mc_test_data.csv")
    #df = pd.read_csv("rndf_filt_data.csv")
    df = pd.read_csv("data.csv")
    #random forest selected the following columns as most predictive
    df = df[['diagnosis','area_worst','concave points_mean','concave points_worst','perimeter_worst','radius_worst']]

    #print(df.head())
    #df = df.drop(["id","Unnamed: 32"],axis=1)
    #df = df.drop(["Unnamed: 0"],axis=1)
    df = df.replace({'diagnosis': "M"}, 1)
    df = df.replace({'diagnosis': "B"}, 0)

    #split dataset for mc seed and testing

    df_mc, df = numpy.split(df, [int(.7*len(df))])

    #split dataset by class
    #df_1 = pd.read_csv("mc_data_M.csv").drop(["Unnamed: 0"],axis=1)
    #df_0 = pd.read_csv("mc_data_B.csv").drop(["Unnamed: 0"],axis=1)
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

    #DTrn =  mc_sim_df['diagnosis']
    #dfTrn = mc_sim_df.drop(['diagnosis'], axis = 1)
    DTrn =  df_mc['diagnosis']
    dfTrn = df_mc.drop(['diagnosis'], axis = 1)
    
    scores = []

    #run model and test
    #randomforest
    model = rfc()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #knn
    model = knc()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #svc
    model = svc(kernel="linear")
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #svc
    model = svc(kernel="rbf")
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #logistic regression
    model = lgr()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    return scores

def classify_mc(s):
    import pandas as pd
    import numpy
    import pandas_montecarlo
    from scipy.stats import shapiro, kruskal, f_oneway
    from sklearn.ensemble import RandomForestClassifier as rfc
    from sklearn.neighbors import KNeighborsClassifier as knc
    from sklearn.svm import SVC as svc
    from sklearn.linear_model import LogisticRegression as lgr

    ## RandomForest Classifier with monte carlo simulated training set
    numpy.random.seed(s)

    #df = pd.read_csv("mc_test_data.csv")
    #df = pd.read_csv("rndf_filt_data.csv")
    df = pd.read_csv("data.csv")
    #random forest selected the following columns as most predictive
    df = df[['diagnosis','area_worst','concave points_mean','concave points_worst','perimeter_worst','radius_worst']]

    #print(df.head())
    #df = df.drop(["id","Unnamed: 32"],axis=1)
    #df = df.drop(["Unnamed: 0"],axis=1)
    df = df.replace({'diagnosis': "M"}, 1)
    df = df.replace({'diagnosis': "B"}, 0)

    #split dataset for mc seed and testing

    df_mc, df = numpy.split(df, [int(.7*len(df))])

    #split dataset by class
    #df_1 = pd.read_csv("mc_data_M.csv").drop(["Unnamed: 0"],axis=1)
    #df_0 = pd.read_csv("mc_data_B.csv").drop(["Unnamed: 0"],axis=1)
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
    
    scores = []

    #run model and test
    #randomforest
    model = rfc()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #knn
    model = knc()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #svc
    model = svc(kernel="linear")
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #svc
    model = svc(kernel="rbf")
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    #logistic regression
    model = lgr()
    model = model.fit(dfTrn.values,DTrn)
    pd = model.predict(dfDev)
    hit = 0
    for i in range(len(pd)):
        if(int(pd[i])==int(DDev.iloc[i])):
            hit+=1
    scores.append(hit/len(pd))
    
    return scores
print('Monte Carlo Simulated Data Classification Testing')
print('Enter number of Simulations(100 or less recommended for first run as this can quite long):')
x = input()

try:
    if(int(x)<3):
        x=3
    x = abs(int(x))
except ValueError:
    #Handle the exception
    print('Error. Please enter an integer')

scoresets = []
scoresets_mc = []
seed = 1000
for i in range(x):
    print("iter: " + str(i+1), end = '\r',flush=True)
    scoresets.append(classify(seed))
    scoresets_mc.append(classify_mc(seed))
    seed+=100

dframe = pd.DataFrame(scoresets,columns=["random_forest",'knn','svc(linear)','svc(rbf)','logistic_regression'])
dframe_mc = pd.DataFrame(scoresets_mc,columns=["random_forest",'knn','svc(linear)','svc(rbf)','logistic_regression'])

import statistics
for col in dframe.columns:
    print("Classifier: ",col)
    print("   Mean Accuracy: Original: ",statistics.mean(dframe[col]))
    print("   Mean Accuracy: M. Carlo: ",statistics.mean(dframe_mc[col]))
    print("   Shapiro pval: Original: ", shapiro(dframe[col])[1])
    print("   Shapiro pval: M. Carlo: ", shapiro(dframe_mc[col])[1])
    if((shapiro(dframe[col])[1]<0.05)and(shapiro(dframe_mc[col])[1]<0.05)):
        print("   ANOVA: ",f_oneway(dframe[col],dframe_mc[col]))
    else:
        print("   Kruskal: ",kruskal(dframe[col],dframe_mc[col]))