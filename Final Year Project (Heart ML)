import pandas as pd
from collections import Counter
import sklearn
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from statsmodels.tools import add_constant as add_constant
import matplotlib.mlab as mlab

# load dataset
heart_data = pd.read_csv("C:/Users/new asus/Downloads/framingham.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# histogram (show distribution)
def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature, color='DarkRed')
        plt.grid(False)
    fig.tight_layout()
    plt.show()

draw_histograms(heart_data, heart_data.columns, 4, 4)

# show number of missing data
print("\n")
print(heart_data.isnull().sum())

# splitting data
new_features = heart_data[["male","age","cigsPerDay","prevalentStroke","sysBP","glucose","TenYearCHD"]]

x = heart_data.iloc[:,:-1]
y = heart_data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=1)

# training data imputation
x_train['education'].fillna(x_train['education'].value_counts().index[0], inplace=True)
x_train['glucose'].fillna(x_train['glucose'].median(), inplace=True)
x_train['cigsPerDay'].fillna(18.35, inplace=True)
x_train['BPMeds'].fillna(x_train['BPMeds'].value_counts().index[0], inplace=True)
x_train['totChol'].fillna(x_train['totChol'].median(), inplace=True)
x_train['BMI'].fillna(x_train['BMI'].median(), inplace=True)
x_train['heartRate'].fillna(x_train['heartRate'].median(), inplace=True)

# testing data imputation
x_test['education'].fillna(x_train['education'].value_counts().index[0], inplace=True)
x_test['glucose'].fillna(x_train['glucose'].median(), inplace=True)
x_test['cigsPerDay'].fillna(18.35, inplace=True)
x_test['BPMeds'].fillna(x_train['BPMeds'].value_counts().index[0], inplace=True)
x_test['totChol'].fillna(x_train['totChol'].median(), inplace=True)
x_test['BMI'].fillna(x_train['BMI'].median(), inplace=True)
x_test['heartRate'].fillna(x_train['heartRate'].median(), inplace=True)

# Undersampling
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1)
X_train_under, y_train_under = undersample.fit_resample(x_train, y_train)

print("Before undersampling: ", Counter(y_train))
print("After undersampling: ", Counter(y_train_under))

# Oversampling
SMOTE = SMOTE(random_state=1)
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(x_train, y_train)

print("Before oversampling: ", Counter(y_train))
print("After oversampling: ",Counter(y_train_SMOTE))


# Logistic Regression
log_reg = sm.Logit(y_train, x_train).fit()
print(log_reg.summary())

logreg=LogisticRegression()
print(logreg.fit(X_train_SMOTE,y_train_SMOTE))
y_pred=logreg.predict(x_test)

# Confusion Matrix (LR)
cm1=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm1,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
plt.title("Confusion Matrix for Logistic Regression")
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm1[0,0]
TP=cm1[1,1]
FN=cm1[1,0]
FP=cm1[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
accuracy = (TP+TN)/(TP+TN+FN+FP)
print("Logistic Regression", "\nAccuracy = ", accuracy, "\nTrue Positive Rate = ", sensitivity, "\nTrue Negative Rate = ", specificity)

# Decision Tree Classifier
d_tree1 = DecisionTreeClassifier(random_state=1)  # by default using Gini Index
d_tree1.fit(X_train_SMOTE, y_train_SMOTE)
y_tree_pred = d_tree1.predict(x_test)

# Feature Importance
from matplotlib import pyplot
importance = d_tree1.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# Confusion matrix (DT)
cm2=confusion_matrix(y_test,y_tree_pred)
conf_matrix=pd.DataFrame(data=cm2,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
plt.title("Confusion Matrix for Decision Tree")
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm2[0,0]
TP=cm2[1,1]
FN=cm2[1,0]
FP=cm2[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
accuracy = (TP+TN)/(TP+TN+FN+FP)
print("\nDecision Tree", "\nAccuracy = ", accuracy, "\nTrue Positive Rate = ", sensitivity, "\nTrue Negative Rate = ", specificity)

# Naive Bayes Classifier
nb1 = GaussianNB()
nb1.fit(X_train_SMOTE, y_train_SMOTE)
y_pred_nb = nb1.predict(x_test)

# Confusion Matrix (NB)
cm3=confusion_matrix(y_test,y_pred_nb)
conf_matrix=pd.DataFrame(data=cm3,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
plt.title("Confusion Matrix for Naive Bayes")
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm3[0,0]
TP=cm3[1,1]
FN=cm3[1,0]
FP=cm3[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
accuracy = (TP+TN)/(TP+TN+FN+FP)
print("\nNaive Bayes", "\nAccuracy = ", accuracy, "\nTrue Positive Rate = ", sensitivity, "\nTrue Negative Rate = ", specificity)

# ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
predictions_dt = d_tree1.predict_proba(x_test)
predictions_log = logreg.predict_proba(x_test)
predictions_nb = nb1.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, predictions_log[:,1])
plt.plot(fpr, tpr, "-b", label="Logistic Regression")
fpr2, tpr2, thresholds2 = roc_curve(y_test, predictions_dt[:,1])
plt.plot(fpr2, tpr2, "-r", label="Decision Tree")
fpr3, tpr3, thresholds3 = roc_curve(y_test, predictions_nb[:,1])
plt.plot(fpr3, tpr3, "-g", label="Naive Bayes")
plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
plt.legend(loc="upper left")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier (downsampled)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
print(roc_auc_score(y_test, predictions_log[:,1]))
print(roc_auc_score(y_test, predictions_dt[:,1]))
print(roc_auc_score(y_test, predictions_nb[:,1]))
plt.grid(True)
plt.show()

