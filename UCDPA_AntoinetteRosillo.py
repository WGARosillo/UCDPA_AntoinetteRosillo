import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
# Using python's operating system to obtain the current working directory##
os.getcwd()
print(os.getcwd())

# changing the current working directory to the path that contains the desired dataset.
os.chdir(r"C:\Users\wgant\OneDrive\Documents\UCDPA_AntoinetteRosillo")
os.getcwd()
print(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

# Invistico Airline is the alias of the airline company that I am investigating. This dataset consists of the
# details of the customers who have already flown with them aims to predict and their feedback on the various
# services and facilities it provides. It aims to predict whether a future customer would be satisfied relative
# to the variety of services the airline provides.This dataset also aims to identify which aspect of their
# service should be prioritized to generate more satisfied customers.##

data = pd.read_csv('Invistico_Airline.csv')
print(data)

data.head()
print(data.head())
print(data.shape)

# The data displays 5 string types and the rest of the data are numerical types.
data.info()

# This used to identify if any of the columns contain missing values. In this case, its the column for
# "Arrival Delay in Minutes" with 393 missing values. Since this does not amount to half of the code, I used the mean
# value you to replace each missing value within the column.
data.isnull().sum()

data['Arrival Delay in Minutes'].describe()

data['Arrival Delay in Minutes']=data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].mean())

data.isnull().sum()
print(data.shape)

# Using a dictionary structure to map the various outcomes of the "satisfaction" column,
# I require a key-value to distinguish the choices "satisfied" and "disatisfied"##
data['satisfaction']= data['satisfaction'].map({'satisfied':1,'dissatisfied':0})
sns.countplot(x="satisfaction", data=data)
plt.title('Airlines Customer satisfaction Count')
plt.xticks([0,1] , ['Dissatisfied',"Satisfied"])
plt.show()

# Visualisation of mapping the "Gender" column, using a dictionary as it is more efficient for mapping.
# This count plot will show the amount of Male and Female customers who flew Invistico, where males are mapped as 0
# and females as 1.
data['Gender']=data['Gender'].map({'Female':1,'Male':0})
sns.countplot(x="Gender", data=data)
plt.title('Airline Gender Count')
plt.xticks([0,1],['Female',"Male"])
plt.show()

# Visualisation of mapping the "Customer Type" column. This count plot shows the amount of Loyal and Disloyal customers
# who flew with Invistico, where Loyal customers were mapped as 1 and Disloyal customers were mapped as 0.
data['Customer Type']=data['Customer Type'].map({'Loyal Customer':1,'disloyal Customer':0})
sns.countplot(x="Customer Type", data=data)
plt.title('Airline Customer Type Count')
plt.xticks([0,1],['Disloyal Customer',"Loyal Customer"])
plt.show()

# Visualisation of mapping the "Type of Travel" column. This count plot shows the type of travel various Invistico
# Airline customers travelled for, one being for business reasons and the other personal reasons.
# This was separated into Business travel and Personal travel, mapped as 1 and 0 respectively
data['Type of Travel']=data['Type of Travel'].map({'Business travel':1,'Personal Travel':0})
sns.countplot(x="Type of Travel", data=data)
plt.title('Airline Travel Type Count')
plt.xticks([0,1],['Personal Travel',"Business travel"])
plt.show()

# Visualisation of mapping the "Class" column. This count plot shows the three types of class options Invistico provides
# their customers, these are Eco, Eco Plus and Business class, mapped as 0, 1 and 2, respectively.
data['Class']=data['Class'].map({'Business':2,'Eco Plus':1,'Eco':0})
sns.countplot(x="Class", data=data)
plt.title('Airline Class Type Count')
plt.xticks([0,1,2],['Eco',"Eco Plus","Business"])
plt.show()

# Visualisation of the relationship of customer satisfaction and the various columns "Gender","Customer Type",
# "Type of Travel" and "Class", referencing their mapping values. Where 1 is satisfied and 0 is dissatisfied.
fig, axs = plt.subplots(2,2,figsize=(14, 14))
fig.subplots_adjust(hspace=0.4,wspace=0.4)
cols=['Gender', 'Customer Type', 'Type of Travel', 'Class']
i=0
for j in range(2):
  for k in range(2):
    sns.countplot(data=data,x=cols[i],hue='satisfaction',ax=axs[j][k])
    axs[j][k].set_xlabel(cols[i])
    axs[j][k].set_ylabel('Customer Satisfaction count')
    axs[j][k].set_title('Customer Satisfaction per {}'.format(cols[i]))
    axs[j][k].legend(['Dissatisfied',"Satisfied"])
    i+=1


# This displays the relationship of the columns "Age" and "satisfaction", allows me to visualize which age displays a
# higher satisfied rate in comparison to dissatisfied customer, where 1 is satisfied and 0 is dissatified.
plt.figure(figsize=(80,15))
ax = sns.countplot(x='Age',hue='satisfaction', data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation= 40, ha = "right")
plt.tight_layout()
plt.title('Customer Satisfaction per Age')
plt.show()

# A variety of count plots dispalying the relationship of customer satisfaction against
# all the other columns.
sns.set(style='white', font_scale=1)
fig = plt.figure(figsize=[40,40])
for i in range(22):
    fig.add_subplot(4, 6, i+1)
    if i in [2, 5, 20, 21]:
        sns.histplot(data=data, x=data.columns[i+1], hue='satisfaction', multiple='stack')
    else:
        sns.countplot(data=data, x=data.columns[i+1], hue='satisfaction')
        sns.despine()
        plt.suptitle('Total Satisfaction', fontsize=20)
        plt.tight_layout()
        fig.savefig('TotalSatisfaction.png')

print(data.corr())

# List of all the numerical columns from the data set including the column satisfaction.
num_list = ["satisfaction", "Age", "Flight Distance", "Seat comfort", "Departure/Arrival time convenient", "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment", "Online support", "Ease of Online booking","On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness", "Online boarding", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
# Using seaborn visualisation tool "heatmap" to portray the correlation between the various numerical type columns.
plt.subplots(figsize=(20,10))
dataplot=sns.heatmap(data[num_list].corr(), annot = True, linewidths= 0.3 , fmt=".1g",cmap= 'viridis')
dataplot.set_xticklabels(dataplot.get_xticklabels(),rotation= 40, ha = "right")
plt.title('Heatmap of the Correlation between Columns')
plt.show()
fig.savefig('heatmap.png')

# due to the correlation values being low with reference to their relationship observed with the satisfaction
# column, the flight distance, Departure/Arrival tim convenient, Gate location, departure Delay in Minutes and Arrival Delay in Minutes columns will be dropped.
data.drop(['Flight Distance','Departure/Arrival time convenient','Gate location','Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1,inplace=True)
print(data.head())
print(data.shape)

# Obtaining training dummies for columns that contains values which are categorised from 0-5, and mapped columns
# as it is necessary to turn them into binary values for further analysis.
training = pd.get_dummies(data, columns = ['Class','Seat comfort','Food and drink', 'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding'])
print(training.head())
print(data.shape)

# Using the scikit learn's train_test_split to split the data set 80% train and 20% test,
# to evaluate my models performance.
from sklearn.model_selection import train_test_split
data_x = training.drop("satisfaction",axis = 1)
data_y = training["satisfaction"]

print(data_x.shape,data_y.shape)
print(data_x.dtypes)

data_x_train,data_x_test, data_y_train, data_y_test = train_test_split(data_x,data_y, test_size=0.2, random_state = 1)
print(data_x_train.shape,data_y_train.shape)
print(data_x_test.shape,data_y_test.shape)

# fit decision tree
from sklearn.tree import DecisionTreeRegressor
data_tree = DecisionTreeRegressor()
data_tree.fit(data_x_train, data_y_train)

# fit random forest
from sklearn.ensemble import RandomForestRegressor
data_forest = RandomForestRegressor(n_jobs=-1)
data_forest.fit(data_x_train, data_y_train)

# fit regression
from sklearn.linear_model import LinearRegression
data_linreg = LinearRegression(n_jobs=-1)
data_linreg.fit(data_x_train, data_y_train)

# Using a tuple to obtain the mean squared error of each classification models, further obtaining the
# Root Mean Squared Error of each of these machine learning algorithms, the lower the value of the
# RMSE the better the model. The lower thr RMSE the better a given model is able to "fit" a dataset.
models= [('linreg', data_linreg), ('random forest', data_forest), ('decision tree', data_tree)]
from sklearn.metrics import mean_squared_error
for i, model in models:
    predictions = model.predict(data_x_train)
    MSE = mean_squared_error(data_y_train, predictions)
    RMSE = np.sqrt(MSE)
    msg = "%s = %.2f" % (i, round(RMSE, 2))
    print('RMSE of', msg)

# RMSE of linreg = 0.30
# RMSE of random forest = 0.07
# RMSE of decision tree = 0.01
# Decision Tree looks very promising as it has the lowest value for RMSE.

# Machine Learning Analysis - Supervised Learning Methods
# Logistic Regression is used to forecast future customer satisfaction with Invistico's
# services and to analyse which service can affect the satisfaction of a customer greatly.
from sklearn.linear_model import LogisticRegression
SEED = 1

logReg = LogisticRegression(random_state= SEED)
logReg.fit(data_x_train,data_y_train)

predictions= logReg.predict(data_x_test)

from sklearn.metrics import accuracy_score
lr_accuracy= accuracy_score(data_y_test,predictions)
print(lr_accuracy)

from sklearn.metrics import  confusion_matrix
confusion_m= confusion_matrix(data_y_test, predictions)
print(confusion_m)

plt.figure(figsize=(5,5))
sns.heatmap(confusion_m, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Reds_r',xticklabels=['Dissatisfied', 'Satisfied'],yticklabels=['Dissatisfied', 'Satisfied'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Logistic Regression Confusion Matrix'
plt.title(all_sample_title, size=15)

# Utilize the outcome of the confusion matrix to analyse the accuracy, the rate of misclassification, precision,
# recall and f1 score of the model generated by logistic regression.

# true_negative = truly dissatisfied customers.
true_negative = confusion_m[0][0]
# false_positive = Customers that were incorrectly labeled as satisfied.
false_positive = confusion_m[0][1]
# false_negative = Customers that were incorrectly labeled as dissatisfied.
false_negative = confusion_m[1][0]
# true_positive = truly satisfied customers.
true_positive = confusion_m[1][1]


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Logistic Regression Accuracy:',Accuracy)
# Accuracy= 0.9012165075454266

misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Logistic Regression misclassification rate:', misclassification_rate)
# Misclasification rate= 0.09878349245457345

Precision = true_positive/(true_positive+false_positive)
print('Logistic Regression Precision:',Precision)
# Precision= 0.9156281335052284

Recall = true_positive/(true_positive+false_negative)
print('Logistic Regression Recall:', Recall)
# Recall= 0.9020604007902907

F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print('Logistic Regression F1 Score:',F1_Score)
# F1 Score= 0.9087936304826899

data_x_train,data_x_test, data_y_train, data_y_test = train_test_split(data_x,data_y, test_size=0.2, random_state=0)

print(data_x_train.shape,data_y_train.shape)
print(data_x_test.shape,data_y_test.shape)

print('Logistic Regression Train Accuracy Score:',logReg.score(data_x_train,data_y_train))
# Logistic Regression Train Accuracy Score: 0.8990029257776408

print('Logistic Regression Test Accuracy Score:',logReg.score(data_x_test,data_y_test))
# Logistic Regression Test Accuracy Score: 0.8977132737911919

# Random Forest Classification is another form of a machine learning algorithm.
from sklearn.ensemble import RandomForestClassifier
data_forest=RandomForestClassifier(random_state=1)
data_forest.fit(data_x_train,data_y_train)
predict_data_forest = data_forest.predict(data_x_test)
predict_data_forest_train = data_forest.predict(data_x_train)

# Depict the accuracy of the random forest model on the test set.
df_score = data_forest.score(data_x_test, data_y_test)
print(df_score)

from sklearn import metrics
cm = metrics.confusion_matrix(data_y_test, predict_data_forest)
print(cm)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=['Dissatisfied', 'Satisfied'], yticklabels=['Dissatisfied', 'Satisfied'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Random Forest Confusion Matrix'
plt.title(all_sample_title, size = 15)

# true_negative = truly dissatisfied customers.
true_negative = cm[0][0]
# false_positive = Customers that were incorrectly labeled as satisfied.
false_positive = cm[0][1]
# false_negative = Customers that were incorrectly labeled as dissatisfied.
false_negative = cm[1][0]
# true_positive = truly satisfied customers.
true_positive = cm[1][1]

# Breaking down the formula for Accuracy which resulted in 0.9513012011087157, which is higher than the
#logistic regression model
rf_accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Random Forest Accuracy: ', rf_accuracy)

rf_misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Random Forest Misclassification rate:', rf_misclassification_rate)
# the Misclassification rate of this model was 0.04869879889128426, which is less than the logistic
# regression model.

# Precison = 0.9617157036511875
rf_Precision = true_positive/(true_positive+false_positive)
print('Random Forest Precision:', rf_Precision)

# Recall = 0.9492652204338698
rf_Recall = true_positive/(true_positive+false_negative)
print('Random Forest Recall:', rf_Recall)

# F1 Score = 0.9087936304826899
rf_F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print('Random Forest F1 Score:', rf_F1_Score)

Feat_Imp = pd.DataFrame({"Features" : data_x_train.columns,"Importance" : data_forest.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'],ascending=False)
print(Feat_Imp)
# The most important feature is Inflight entertainment followed by seat comfort, the least important feature is
# Onboard service.

plt.figure(figsize = (10,50))
rf=sns.barplot(x="Importance",y="Features",data = Feat_Imp)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print('Train Accuracy Score:',data_forest.score(data_x_train,data_y_train))
# Random Forest Train Accuracy Score: 0.9997978903603326

print('Test Accuracy Score:',data_forest.score(data_x_test,data_y_test))
# Random Forest Test Accuracy Score: 0.9513012011087157

from sklearn.metrics import classification_report
print('\n classification report:\n',classification_report(data_y_test,predict_data_forest))

# Random Forest Regressor is used when carrying out hyperparameter tuning for Random Forest Classifier.
# Use GridSearchCV as the hyperparameter tuning method, as it carries out a cross validation on
# all the combinations of hyperparameters that are set. Investigate the major drawback when compared
# RandomizedSearchCV which is that it is computationally very expensive and can take twice as long to run
# depending on the dataset.
from sklearn.ensemble import RandomForestRegressor

Seed=1

rf_reg= RandomForestRegressor(random_state=SEED)

rf_reg.get_params()

from  sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

params_rf= {'n_estimators':[100,200,300], 'max_depth':[4,6,8], 'min_samples_leaf':[0.1,0.2,0.3], 'max_features':['auto']}

grid_rf_reg= GridSearchCV(estimator= rf_reg, param_grid= params_rf, cv= 3, scoring='neg_mean_squared_error',verbose=1,
                          n_jobs=-1)
grid_rf_reg.fit(data_x_train,data_y_train)

# Extract the best hyperparameters for Random Forest
best_rf_reg= grid_rf_reg.best_params_
print('Best hyperparameters:\n', best_rf_reg)

best_model = grid_rf_reg.best_estimator_
best_model_rf_reg= best_model.predict(data_x_test)
mse_rfr=MSE(data_y_test,best_model_rf_reg)

# Obtain the Root Mean Square Error of the random forest test model.
rmse_rfr= mse_rfr**(1/2)
print('Random Forest Regression Test set RMSE: {:.2f}'.format(rmse_rfr))
# Random Forest Regression Test set RMSE: 0.39.


# Decision Tree is another form of a Machine Learning algorithm.
from sklearn.tree import DecisionTreeClassifier
data_tree = DecisionTreeClassifier(random_state=1)
data_tree.fit(data_x_train, data_y_train)

predictions = data_tree.predict(data_x_test)

# The accuracy of the Decision Tree Model test set.
dt= data_tree.score(data_x_test, data_y_test)
print('Decision Tree Accuracy',dt)
# Decision Tree Accuracy 0.9357098860486603


from sklearn.metrics import confusion_matrix
dt_cm = confusion_matrix(data_y_test,predictions)
print(dt_cm)

plt.figure(figsize=(5,5))
sns.heatmap(dt_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Greens_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Decision Tree Confusion Matrix'
plt.title(all_sample_title, size = 15)

# true_negative = truly dissatisfied customers.
true_negative = dt_cm[0][0]
# false_positive = Customers that were incorrectly labeled as satisfied.
false_positive = dt_cm[0][1]
# false_negative = Customers that were incorrectly labeled as dissatisfied.
false_negative = dt_cm[1][0]
# true_positive = truly satisfied customers.
true_positive = dt_cm[1][1]

# Breaking down the formula for Accuracy
dt_Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Decision Tree Accuracy:', dt_Accuracy)
#Decision Tree Accuracy: 0.9357098860486603

dt_misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Decision Tree Misclassification rate:', dt_misclassification_rate)
#Decision Tree Misclassification rate: 0.0642901139513397

# Precison
dt_Precision = true_positive/(true_positive+false_positive)
print('Decision Tree Precision:',dt_Precision)
# Decision Tree Precision: 0.9403964265773311

# Recall
dt_Recall = true_positive/(true_positive+false_negative)
print('Decision Tree Recall:', dt_Recall)
#Decision Tree Recall: 0.9428971308607418

#F1 Score
dt_F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print('Decision Tree F1 Score:', dt_F1_Score)
#Decision Tree F1 Score: 0.9087936304826899

print('Train Accuracy Score:',data_tree.score(data_x_train,data_y_train))
#Train Accuracy Score: 0.9997978903603326

print('Test Accuracy Score:',data_tree.score(data_x_test,data_y_test))
#Test Accuracy Score: 0.9357098860486603

from sklearn.metrics import classification_report
print('\n classification report:\n',classification_report(data_y_test,predictions))

Feature_Imp = pd.DataFrame({"Features" : data_x_train.columns,"Importance" : data_tree.feature_importances_})
Feature_Imp = Feature_Imp.sort_values(by=['Importance'],ascending=False)
print(Feature_Imp)
# The most important feature is still Inflight entertainment and still followed by seat comfort, but the
# least important feature for the decision tree model is ease of online booking.

plt.figure(figsize=(10, 50))
loca = sns.barplot(x="Importance",y="Features",data = Feature_Imp)
sns.barplot(x="Importance", y="Features", data=Feature_Imp)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.show()

# Hyperparameter Tuning a Decision Tree to obtain an optimal model structure resulting in the best model performance.
# Use the cross validation method to estimate the generalization performance and root mean squared error.
from sklearn.model_selection import RandomizedSearchCV
criterion = ['gini','entropy']
splitter = ['best','random']
max_depth = [6,10,15,20]
min_samples_split = [5,7,10]
min_samples_leaf = [0.1,0.5,1]

hyperparams = dict(criterion=criterion, splitter= splitter, max_depth= max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)

data_tree_tuning = RandomizedSearchCV(data_tree, hyperparams, cv=5, scoring='f1')
data_tree_tuning.fit(data_x_train,data_y_train)

prediction = data_tree_tuning.predict(data_x_test)
accuracy= accuracy_score(data_y_test,prediction)
print(accuracy)

print('Best criterion:', data_tree_tuning.best_estimator_.get_params()['criterion'])
print('Best splitter:', data_tree_tuning.best_estimator_.get_params()['splitter'])
print('Best max_depth:', data_tree_tuning.best_estimator_.get_params()['max_depth'])
print('Best min_samples_split:', data_tree_tuning.best_estimator_.get_params()['min_samples_split'])
print('Best min_samples_leaf:', data_tree_tuning.best_estimator_.get_params()['min_samples_leaf'])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error as MSE
data_x_train,data_x_test,data_y_train,data_y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=1)
data_tree_best=DecisionTreeClassifier(criterion='entropy',max_depth=20,min_samples_split=5,min_samples_leaf=1,random_state=1,splitter='random')
data_tree_best.fit(data_x_train,data_y_train)
pred_dtb=data_tree_best.predict(data_x_test)
mse_dtb= MSE(data_y_test,pred_dtb)
rmse_dtb= mse_dtb**(1/2)
print('Decision Tree Root Mean Square Error: ',rmse_dtb)
# Decision Tree Root Mean Square Error:  0.23314774028922317
dtb_cm = confusion_matrix(data_y_test,pred_dtb)
print('Best Decision Tree Confusion matrix:',dtb_cm)

# true_negative = truly dissatisfied customers.
true_negative = dtb_cm[0][0]
# false_positive = Customers that were incorrectly labeled as satisfied.
false_positive = dtb_cm[0][1]
# false_negative = Customers that were incorrectly labeled as dissatisfied.
false_negative = dtb_cm[1][0]
# true_positive = truly satisfied customers.
true_positive = dtb_cm[1][1]

# Breaking down the formula for Accuracy
dtb_Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Best Decision Tree Accuracy:', dtb_Accuracy)
# Best Decision Tree Accuracy: 0.945642131198029

dtb_misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
print('Best Decision Tree Misclassification rate:', dtb_misclassification_rate)
# Best Decision Tree Misclassification rate: 0.05435786880197105

# Precison
dtb_Precision = true_positive/(true_positive+false_positive)
print('Best Decision Tree Precision:',dtb_Precision)
# Best Decision Tree Precision: 0.959389400921659

# Recall
dtb_Recall = true_positive/(true_positive+false_negative)
print('Best Decision Tree Recall:', dtb_Recall)
# Best Decision Tree Recall: 0.9401637030764889

# F1 Score
dtb_F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print('Best Decision Tree F1 Score:', dtb_F1_Score)
# Best Decision Tree F1 Score: 0.9087936304826899

print('Train Accuracy Score:',data_tree_best.score(data_x_train,data_y_train))
#Train Accuracy Score: 0.9722628580227902

print('Test Accuracy Score:',data_tree_best.score(data_x_test,data_y_test))
# Test Accuracy Score: 0.945642131198029

print('\n classification report:\n',classification_report(data_y_test,pred_dtb))

# Decision Tree Regressor cross validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

data_x_train,data_x_test,data_y_train,data_y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=1)
dtr_cv=DecisionTreeRegressor(max_depth=20,min_samples_split=5,min_samples_leaf=1,random_state=1)
dtr_cv.fit(data_x_train,data_y_train)
predict_dtr_train=dtr_cv.predict(data_x_train)
predict_dtr_test=dtr_cv.predict(data_x_test)

MSE_CV = - cross_val_score(dtr_cv,data_x_train,data_y_train,cv=10,scoring='neg_mean_squared_error',n_jobs = -1)
RMSE_CV=(MSE_CV.mean())**(1/2)

print('Decision Tree Cross Validation RMSE:{:.2f}'.format(RMSE_CV))
# Decision Tree Cross Validation RMSE:0.23

RMSE_train=(MSE(data_y_train,predict_dtr_train))**(1/2)
print('Decision Tree Train RMSE:{:.2f}'.format(RMSE_train))
# Decision Tree Train RMSE:0.13

# Comparing the Root Mean Squared Error of each of the regression methods associated with Logistic Regression,
# Decision Tree and Random Forest.

from sklearn.tree import DecisionTreeRegressor
dt_regressor=DecisionTreeRegressor(max_depth=20,min_samples_split=5,min_samples_leaf=1,random_state=1)
dt_regressor.fit(data_x_train,data_y_train)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=1)
logreg.fit(data_x_train,data_y_train)

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=1, max_depth=4,max_features='auto',min_samples_leaf=0.1,n_estimators=100)
rf_regressor.fit(data_x_train,data_y_train )

# Use a tuple to obtain the mean squared error of each classifiaction model.
classifiers=[('Logistic Regression',logreg), ('Random Forest Regressor',rf_regressor), ('Decision Tree Regressor',dt_regressor)]
from sklearn.metrics import mean_squared_error
for i, classifier in classifiers:
    predictions = classifier.predict(data_x_train)
    MSE = mean_squared_error(data_y_train, predictions)
    RMSE = np.sqrt(MSE)
    msg = "%s = %.2f" % (i, round(RMSE, 2))
    print('RMSE of', msg)

# RMSE of Logistic Regression = 0.32
# RMSE of Random Forest Regressor = 0.38
# RMSE of Decision Tree Regressor = 0.13
# The random forest model displayed a high accuracy rate out of the machine learning algorithms at 0.99 and 0.94,
# for its training and test set respectively. It also showed the highest value for RMSE at 0.38. Decision Tree had a
# higher accuracy rate in comparison to the Random Forest method, at 0.99 and 0.95, for their train set and test set,
# respectively. The Decision Tree model also has the lowest value for RMSE at 0.13, therefore
# I would recommend to use the decision tree classification model to analyse this data set.

# To display that two data frames can be merged, create and use a dummies, to form a combined
# data set.
df1= pd.DataFrame({
    "coffee":["iced latte", "cappucino", "flat white"],
    "price per drink":["3.85","3.40","3.40"]
})
print(df1)

df2= pd.DataFrame({
    "coffee":["iced latte","cappucino","flat white"],
    "milk used":["skimmed milk","low fat milk","full fat milk"]
})
print(df2)

coffee = pd.merge(df1,df2,on="coffee")
# Inner join by default
print(coffee)
