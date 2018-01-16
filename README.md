# Spark-AWS-Scala-Predict
Dataset: R -dataset ,Mortality Outcomes for Females Suffering Myocardial Infarction.

FEATURE SELECTION : Correlation was used to determine relevant features. Yronset features is the least related column with 0.08
MODELS IMPLEMENTED : Random Forest, Logistic Regression, Linear SVM , Decision Tree , Gradient Boosting Tree, Naive Bayes       
ANALYSIS:
Random forest and Naive Bayes, gives the best accuracy of 81% , all the models are approximately around 78%Linear SVM is the 
least accurate among the all models with the default setMaxitertor-10.Linear SVM  gives 0 false Positive for some testing 
samples, which is inaccurate, thus we have applied different hyperparameters and found setMaxitertor= 20 gives best 
performance in Linear SVM.
Also divided age groups and found that least survival rate was found in elderly ladies(60-above) and best survival rate was found in middle aged group 
(40-50)
