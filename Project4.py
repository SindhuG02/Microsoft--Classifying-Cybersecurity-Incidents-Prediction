import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
#Since data is huge we can't use KNN
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
  # type: ignore

warnings.filterwarnings('ignore') 

max_depth_value=0
min_samples_split_value=0
#read Train data file
df_Train=pd.read_csv("Project4/GUIDE_Train.csv")

#it's having impure data
#BenignPositive    4110817
#TruePositive      3322713
#FalsePositive     2031967
#df_Train['IncidentGrade'] = df_Train['IncidentGrade'].astype('category')

#print(df_Train['IncidentGrade'].unique())
#Target column to replace values in the form of number for ML to work
df_Train["IncidentGrade"]=df_Train["IncidentGrade"].replace('BenignPositive',0)
df_Train["IncidentGrade"]=df_Train["IncidentGrade"].replace('FalsePositive',1)
df_Train["IncidentGrade"]=df_Train["IncidentGrade"].replace('TruePositive',2)


One_Hot_Encoder = OneHotEncoder() 
#got erro in test data so droping
df_Train=df_Train.drop('Timestamp',axis=1)
df_Train=df_Train.drop(['ThreatFamily','ActionGranular','LastVerdict','Id','OrgId','EmailClusterId','MitreTechniques','SuspicionLevel'],axis=1)

#print("Check for null values - before",df_Train.isna().sum())
df_Train=df_Train.ffill()
df_Train=df_Train.bfill()

print("Check for null values - After",df_Train.isna().sum())
df_Train=pd.get_dummies(df_Train, columns = ['Category','ActionGrouped','EntityType','EvidenceRole','ResourceType','Roles','AntispamDirection'],dtype='float')

X_train=df_Train.drop("IncidentGrade",axis=1)
Y_train=df_Train['IncidentGrade']
Randome_model=RandomUnderSampler()
print("value",Y_train.value_counts())
New_X_Train,New_Y_Train=Randome_model.fit_resample(X_train,Y_train)
print("new value",New_Y_Train.value_counts())

#read Test data file
df_Test=pd.read_csv("Project4/GUIDE_Test.csv")

#print(df_Test['IncidentGrade'].unique())
#Target column to replace values in the form of number for ML to work
df_Test["IncidentGrade"]=df_Test["IncidentGrade"].replace('BenignPositive',0)
df_Test["IncidentGrade"]=df_Test["IncidentGrade"].replace('FalsePositive',1)
df_Test["IncidentGrade"]=df_Test["IncidentGrade"].replace('TruePositive',2)
df_Test=df_Test.drop(['ThreatFamily','ActionGranular','LastVerdict','Id','OrgId','Usage','MitreTechniques','SuspicionLevel'],axis=1)
df_Test=df_Test.drop('Timestamp',axis=1)
df_Test=df_Test.drop('EmailClusterId',axis=1)

#numpy._core._exceptions._ArrayMemoryError: Unable to allocate 981. MiB for an array with shape (31, 4147992) and data type int64
#print("Check for null values - before",df_Train.isna().sum())
df_Test=df_Test.ffill()
df_Test=df_Test.bfill()

df_Test=pd.get_dummies(df_Test, columns = ['Category','ActionGrouped','EntityType','EvidenceRole','ResourceType','Roles','AntispamDirection'],dtype='float')

print("Check for null values - After",df_Test.isna().sum())

X_test=df_Test.drop("IncidentGrade",axis=1)
Y_test=df_Test['IncidentGrade']
print(Y_test.value_counts())
print(X_test.info())
New_X_Test,New_Y_Test=Randome_model.fit_resample(X_test,Y_test)
print(New_X_Test.info())
print("new value for test",New_Y_Test.value_counts())


#print("Check for null values - After",df_Test.isna().sum())
#print(df_Test.info())
#print("----------------------")
#print(df_Train.info())
'''
params={"max_depth":np.arange(4,20),"min_samples_split":np.arange(10,100)}
print("Processing.")
model=DecisionTreeClassifier(random_state=42)
print("Processing.....")
model=GridSearchCV(estimator=model,param_grid=params)
print("Processing...........")
model.fit(New_X_Train,New_Y_Train)
print("Processing..................")
print(model.best_params_)
maxDepth_and_minSamples_values=model.best_params_
print(maxDepth_and_minSamples_values.values())
print(type(model.best_params_))
max_depth_value=maxDepth_and_minSamples_values.get("max_depth")
min_samples_split_value=maxDepth_and_minSamples_values.get("min_samples_split")
print("min_samples_split_value",min_samples_split_value)
print("max_depth_value",max_depth_value)

'''
models=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier()]
for model in models:
    print(model)
    model.fit(New_X_Train,New_Y_Train)
    
    train_predict=model.predict(New_X_Train)

    model.fit(New_X_Test,New_Y_Test)
    test_predict=model.predict(New_X_Test)

    print("************* Train ************")
    print("Train confusion_matrix",confusion_matrix(New_Y_Train,train_predict))
    print("Train accuracy_score",accuracy_score(New_Y_Train,train_predict))
    print("Train precision_score",precision_score(New_Y_Train,train_predict,average='micro'))
    print("Train Recall_Score",recall_score(New_Y_Train,train_predict,average='micro'))
    print("Train f1_score",f1_score(New_Y_Train,train_predict,average='micro'))
    
    
    print("************* Test ************")
    print("Test confusion_matrix",confusion_matrix(New_Y_Test,test_predict))
    print("Test accuracy_score",accuracy_score(New_Y_Test,test_predict))
    print("Test precision_score",precision_score(New_Y_Test,test_predict,average='micro'))
    print("Test Recall_Score",recall_score(New_Y_Test,test_predict,average='micro'))
    print("Test f1_score",f1_score(New_Y_Test,test_predict,average='micro'))
    
    print("******************************")
    
