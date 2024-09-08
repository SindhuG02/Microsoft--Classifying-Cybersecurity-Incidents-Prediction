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


Results:

    
LogisticRegression()
    
************* Train ************

Train accuracy_score 0.5102689157840568 

Train precision_score 0.5102689157840568

Train Recall_Score 0.5102689157840568

Train f1_score 0.5102689157840568

************* Test ************

Test accuracy_score 0.5284613458764725

Test precision_score 0.5284613458764725

Test Recall_Score 0.5284613458764725

Test f1_score 0.5284613458764725

******************************

   DecisionTreeClassifier()
   
************* Train ************

Train accuracy_score 0.9999761787520342

Train precision_score 0.9999761787520342

Train Recall_Score 0.9999761787520342

Train f1_score 0.9999761787520342

************* Test ************

Test accuracy_score 1.0

Test precision_score 1.0

Test Recall_Score 1.0

Test f1_score 1.0

******************************

   RandomForestClassifier()
   
************* Train ************

Train accuracy_score 0.9999720997712181

Train precision_score 0.9999720997712181

Train Recall_Score 0.9999720997712181

Train f1_score 0.9999720997712181

************* Test ************

Test accuracy_score 0.9999951995757902

Test precision_score 0.9999951995757902

Test Recall_Score 0.9999951995757902

Test f1_score 0.9999951995757902

******************************


