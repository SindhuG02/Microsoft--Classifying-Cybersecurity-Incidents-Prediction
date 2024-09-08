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
step 6:- best approach will be to find the max_depth & min_samples_split for the decisionTress but since data is hug it will run for a day. once we get the data we can pass that withing parameters for the Decision           Tree
step 7:- check for all the models and fit the model and check the values for each test n train files.
step 8:- pick the model which is having the better performance.
