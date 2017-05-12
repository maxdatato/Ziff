
# coding: utf-8

# In[2]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('churn.csv',index_col=0)
df =df.replace('?','0')
df.head()


# In[2]:

df.columns


# In[3]:

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Isolate target data
churn_result = df['Churn?']
y = np.where(churn_result == 'True.',1,0)
# y = df.iloc[:, 20].values
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# df['Churn'] = (df['Churn?'] == 'True.')
# y = df['Churn'].as_matrix().astype(np.int)
print('There are {} instances for churn class and {} instances for not-churn classes.'.format(y.sum(), y.shape[0] - y.sum()))
print('Ratio of churn class over all instances: {:.2f}'.format(float(y.sum()) / y.shape[0]))


# In[ ]:




# In[4]:

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df2 = df

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns
X = churn_feat_space.as_matrix().astype(np.float)

# State is string and we want discre integer values
# labelencoder = LabelEncoder()
# df2['State'] = labelencoder.fit_transform(df2['State'])
# Drop the redundant columns from dataframe
# df2.drop(['Area Code','Phone','Churn','Churn?'], axis=1, inplace=True)
# Get the features as integers similar to what we did for labels(targets)
# df2[["Int'l Plan","VMail Plan"]] = df2[["Int'l Plan","VMail Plan"]] == 'yes'
# df2.head(100)

# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features = [1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]


# In[ ]:




# In[5]:

# This is important
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature space holds %d observations and %d features", X.shape)
print ("Unique target labels:", np.unique(y))


# In[ ]:




# In[ ]:




# In[ ]:




# In[6]:

from sklearn.model_selection import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


# In[7]:

from sklearn.model_selection import StratifiedKFold

def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


# In[14]:

from sklearn.metrics import accuracy_score as accuracy
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_score

kf = KFold(len(y),n_folds=10,shuffle=True)
kf2 = StratifiedKFold(y,n_folds=10,shuffle=True)

print('Dump Classifier:               {:.3f}'.format(accuracy(y, [0 for ii in y.tolist()])))

print('Logistic Regression:           {:.3f}'.format(accuracy(y, run_cv(X, y, linear_model.LogisticRegression))))
print('Logistic Regression:           {:.3f}'.format(accuracy(y, stratified_cv(X, y, linear_model.LogisticRegression))))

print('K Nearest Neighbor Classifier: {:.3f}'.format(accuracy(y, run_cv(X, y, neighbors.KNeighborsClassifier))))
print('K Nearest Neighbor Classifier: {:.3f}'.format(accuracy(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))

print('Support vector machine(SVM):   {:.3f}'.format(accuracy(y, run_cv(X, y, svm.SVC))))
print('Support vector machine(SVM):   {:.3f}'.format(accuracy(y, stratified_cv(X, y, svm.SVC))))

print('Decision Tree Classifier:      {:.3f}'.format(accuracy(y, run_cv(X, y, tree.DecisionTreeClassifier))))
print('Decision Tree Classifier:      {:.3f}'.format(accuracy(y, stratified_cv(X, y, tree.DecisionTreeClassifier))))

print('Random Forest Classifier:      {:.3f}'.format(accuracy(y, run_cv(X, y, ensemble.RandomForestClassifier))))
print('Random Forest Classifier:      {:.3f}'.format(accuracy(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
results = cross_val_score(ensemble.RandomForestClassifier(), X = X, y = y, cv = kf2)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

print('Gradient Boosting Classifier:  {:.3f}'.format(accuracy(y, run_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Gradient Boosting Classifier:  {:.3f}'.format(accuracy(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
results = cross_val_score( ensemble.GradientBoostingClassifier(), X = X, y = y, cv = kf2)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

print('XGBoost:                       {:.3f}'.format(accuracy(y, run_cv(X, y, XGBClassifier))))
print('XGBoost:                       {:.3f}'.format(accuracy(y, stratified_cv(X, y, XGBClassifier))))
results = cross_val_score( XGBClassifier(), X = X, y = y, cv = kf2)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




# In[ ]:




# In[ ]:




# In[9]:

from sklearn.metrics import confusion_matrix 

dumb_conf_matrix = confusion_matrix(y, [0 for ii in y.tolist()]); # ignore the warning as they are all 0
logistic_reg_conf_matrix = confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
k_neighbors_conf_matrix = confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
svm_svc_conf_matrix = confusion_matrix(y, stratified_cv(X, y, svm.SVC))
decision_conf_matrix = confusion_matrix(y, stratified_cv(X, y, tree.DecisionTreeClassifier))
random_forest_conf_matrix = confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
grad_ens_conf_matrix = confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
XGB_ens_conf_matrix = confusion_matrix(y, stratified_cv(X, y, XGBClassifier))


# In[10]:

from sklearn.metrics import classification_report 

print('Dump Classifier:\n {}\n'.format(classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
print('Logistic Regression:\n {}\n'.format(classification_report(y, stratified_cv(X, y, linear_model.LogisticRegression))))
print('Support vector machine(SVM):\n {}\n'.format(classification_report(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:\n {}\n'.format(classification_report(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier:\n {}\n'.format(classification_report(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Gradient Boosting Classifier:\n {}\n'.format(classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))


# In[11]:

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# sc_y = StandardScaler()
# y = sc_y.fit_transform(y)

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# y_train = sc_X.fit_transform(y_train)


# In[12]:

gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)
# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



