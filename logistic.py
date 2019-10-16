# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:18:34 2019

@author: Anuja
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("E:\\analytics\\python\\class")
dataset = pd.read_csv("data.csv")
dataset.describe()
dataset.shape
dataset.shape[0]
dataset.shape[1]
list(dataset)
dataset.info()
##based on gre gpa rank , kid will et admission or not
#getting admission = 1
#not getting admission = 0
##Reference Category
from patsy import dmatrices,Treatment
y,X = dmatrices("admit~gre + gpa+ C(rank,Treatment(reference = 4))",dataset,return_type = "dataframe")
##admit is taken as dependent and rank astreated as dummy variable wrh referrence category 4
##C combine
##~ equilence
y.shape
X.shape
X.head()
##split data into two parts
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
## same dataset pe jayenge for random_state
##building logistic
#fit Logit model
import statsmodels.api as sm
logit = sm.Logit(y_train, X_train)
result = logit.fit()
##summary of LR
result.summary()
result.params
##confusion matrix and odd ratio
cnf_matrix = result.pred_table()
cnf_matrix
##odd ratio
import numpy as np
np.exp(result.params)##got exponential
##Prediction on Test data
y_pred = result.predict(X_test)
from sklearn import metrics
##import metrices class
##auc's curve  =roc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
##calculate area under curve
##auc on test data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
auc(false_positive_rate, true_positive_rate)
##bekar auc, should be higher than 0.5
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
class_names = [0,1]## name of classes
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot = True,cmap = "YlGnBu",fmt = "g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y = 10.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
Text(0.5,257.44,"Predicted label")##not working
##Code for confusion matrix
##LR using sklearn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred1 = logreg.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred1))
print("Recall:", metrics.recall_score(y_test,y_pred1))





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.getcwd()
os.chdir("E:\\analytics\\python\\class")
datadoc = pd.read_csv("data.csv")
datadoc.shape
datadoc.shape[0]
datadoc.describe()
datadoc.info()
from patsy import dmatrices,Treatment
y, x = dmatrices("admit~gre+gpa+C(rank,Treatment(reference = 4))", datadoc,return_type="dataframe")
y.shape
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
y_train.shape
import statsmodels.api as sm
logit = sm.Logit(y_train, x_train)
result = logit.fit()
result.summary()
result.params
cnf_matrix = result.pred_table()
np.exp(result.params)
y_pred = result.predict(x_test)
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
false_positive_rate,true_positive_rate, threshold = roc_curve(y_test,y_pred)
auc(false_positive_rate,true_positive_rate)
import seaborn as sns
%matplotlib inline
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(cnf_matrix),annot = True,cmap ="YlGnBu",fmt = "g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y = 1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

##Another process using sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred1 = logreg.predict(x_test)
print("Accuracy",metrics.accuracy_score(y_test,y_pred1))
precision = print("Precision",metrics.precision_score(y_test,y_pred1))
recall = print("Recall",metrics.recall_score(y_test,y_pred1))
f1_score = 2*((recall+precision)/(recall*precision))
cnf_matrix1 = metrics.confusion_matrix(y_test,y_pred1)
cnf_matrix1
import seaborn as sns
%matplotlib inline
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(cnf_matrix),annot = True,cmap ="YlGnBu",fmt = "g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y = 1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
##Roc and AUC
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test,y_pred_proba)
gini = 2*auc-1
gini
plt.plot(fpr,tpr,label = "data1,auc="+str(auc))
plt.legend
plt.show()
##KS
Test_data1 = pd.concat([y_test,y_pred],axis = 1)
Test_data1.columns = ["dep_flag","prob"]
Test_data1.columns
Test_data1["decile"] = pd.qcut(Test_data1["prob"],10,labels = ['1','2','3','4','5','6','7','8','9','10'])
Test_data1.head()
Test_data1.columns  = ["Event","Probability","Decile"]
Test_data1.head()
Test_data1["NonEvent"] = 1-Test_data1["Event"]
Test_data1.head()
df1 = pd.pivot_table(data=Test_data1,index=["Decile"],values = ["Event","NonEvent","Probability"],aggfunc = {"Event" :[np.sum],"NonEvent":[np.sum], "Probability" : [np.min,np.max]})
df1.head()
df1.reset_index()
df1.columns=["Event_count","NonEvent_count","max_score","min_score"]
df1.head()
df1["Total_cust"] = df1["Event_count"]+df1["NonEvent_count"]
df1
df2 = df1.sort_values(by = "min_score",ascending= False)
df2
df2["Event_rate"] = (df2["Event_count"]/df2["Event_count"]).apply( "{0:.2%}".format)
df2.head()
event_sum = df2["Event_count"].sum()
nonevent_sum= df2["NonEvent_count"].sum()
df2["Event %"]=(df2["Event_count"]/event_sum).apply( "{0:.2%}".format)
df2["nonevent %"]=(df2["NonEvent_count"]/nonevent_sum).apply( "{0:.2%}".format)
df2["KS_stats"] = np.round(((df2["Event_count"]/df2["Event_count"].sum()).cumsum())-(df2["NonEvent_count"]/df2["NonEvent_count"].sum()).cumsum(),4) * 100
df2
flag = lambda x: "*****" if x == df2["KS_stats"].max() else " " 
df2["max_ks"] = df2["KS_stats"].apply(flag)
df2.to_csv("KS_test.csv")
##Gains Chart
df_test1 = df2.copy() 
df_test1["Event_cum%"] = np.round(((df_test1["Event_count"]/df_test1["Event_count"].sum()).cumsum()),4)*100
df_test1
df_test2 = df_test1[["Event_cum%"]]
df_test2.reset_index()
df_test2.columns = ["Event_cum%_test"]
df_test2
df_test2["Base_%"] = [10,20,30,40,50,60,70,80,90,100]
df_test2
gains_chart = df_test2.plot(kind ="line",use_index = False)
gains_chart.set_ylabel("Proportion of Event")
gains_chart.set_xlabel("Decile")
gains_chart.set_title("Gains Chart")

##Lift Chart
final = df_test2.copy()
final["Lift_test"]= (df_test2["Event_cum%_test"]/df_test2["Base_%"]) 
final["Baseline"] = [1,1,1,1,1,1,1,1,1,1]
Lift_chart = final[["Lift_test","Baseline"]]
Lift_graph = Lift_chart .plot(kind = "line",use_index = False)
Lift_graph.set_ylabel("Lifts")
Lift_graph.set_xlabel("Decile")
Lift_graph.set_title("Lift_graph")
lift_graph.set_ylim(0.0,2)













