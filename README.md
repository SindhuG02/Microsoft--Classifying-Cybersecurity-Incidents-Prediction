# Microsoft--Classifying-Cybersecurity-Incidents-Prediction

the data is classification data so we have to use the KNN(),DecisionTree(),Logistic Regression(),RandomForestClassifier(),GradientBoostingClassifier()

step 1:- load the data into pandas

step 2:- remove the unnecessary columns

step 3:- remove the null values, there are few ways to remove the null values.
          -> delete the row
          -> bfill - backward fill
          -> ffill - forward fill
          -> replace with mean/median values.

          But in our case we are using the bfill & ffill for removing the null values.

step 4:- Since ML will understand only the number we have to convert all the columns with string to number. we can do that using oneHot encoding/label encoding
          -> we are using the OneHotEncoding because the data we have does not follow the order.(it's equal priority).

step 5:- perform the same steps for the TEST data also becasue we have the different csv file for train n test data.

step 6:- best approach will be to find the max_depth & min_samples_split for the decisionTress but since data is hug it will run for a day. once we get the data we can pass that withing parameters for the DecisionTree

step 7:- check for all the models and fit the model and check the values for each test n train files.

step 8:- pick the model which is having the better performance.

Based on results RandomForestClassifier() is the best option.
Results:

Results:
    LogisticRegression()

Train confusion_matrix 
[[1034057  361593  647344]
 [ 610290  616716  815988]
 [ 377779  182497 1482718]]

Train accuracy_score 0.5112579870523359
Train precision_score 0.5112579870523359
Train Recall_Score 0.5112579870523359
Train f1_score 0.5112579870523359

************* Test ************

Test confusion_matrix 
[[461144 165372 276182]
 [246839 304911 350948]
 [158617  66979 677102]]

Test accuracy_score 0.5329050616411395
Test precision_score 0.5329050616411395
Test Recall_Score 0.5329050616411395
Test f1_score 0.5329050616411395

******************************

DecisionTreeClassifier()

************* Train ************

Train confusion_matrix 
[[2042994       0       0]
 [     64 2042929       1]
 [     59      52 2042883]]

Train accuracy_score 0.9999712839750549
Train precision_score 0.9999712839750549
Train Recall_Score 0.9999712839750549
Train f1_score 0.9999712839750549

************* Test ************

Test confusion_matrix [[902698      0      0]
 [     0 902698      0]
 [     0      0 902698]]

Test accuracy_score 1.0
Test precision_score 1.0
Test Recall_Score 1.0
Test f1_score 1.0

******************************

RandomForestClassifier()

************* Train ************

Train confusion_matrix 
[[2042927      40      27]
 [     33 2042929      32]
 [     40      37 2042917]]

Train accuracy_score 0.9999658997203777
Train precision_score 0.9999658997203777
Train Recall_Score 0.9999658997203777
Train f1_score 0.9999658997203777

************* Test ************

Test confusion_matrix
[[902692      5      1]
 [     4 902694      0]
 [     2      3 902693]]

Test accuracy_score 0.9999944610489887
Test precision_score 0.9999944610489887
Test Recall_Score 0.9999944610489887
Test f1_score 0.9999944610489887

******************************

GradientBoostingClassifier()

************* Train ************

Train confusion_matrix 
[[1775430  153432  114132]
 [ 576110 1351373  115511]
 [ 470638  113419 1458937]]

Train accuracy_score 0.7482058194982462
Train precision_score 0.7482058194982462
Train Recall_Score 0.7482058194982462
Train f1_score 0.7482058194982462

************* Test ************

Test confusion_matrix 
[[791133  78057  33508]
 [232609 624615  45474]
 [202576  39555 660567]]

Test accuracy_score 0.7667071379353892
Test precision_score 0.7667071379353892
Test Recall_Score 0.7667071379353892
Test f1_score 0.7667071379353892

******************************

