import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score


import warnings
warnings.filterwarnings("ignore")

def detect_outliers(column,df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def grid_search_run(param_grid,model):
    grid_search = GridSearchCV(model, param_grid, cv=7, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search

pd.set_option('display.max_columns', None)

data = pd.read_csv("D:/Coding Projects/Breast Cancer Classification AI/Dataset/data.csv")

data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})


data = detect_outliers('radius_mean',data)
data = detect_outliers('texture_mean',data)
data = detect_outliers('perimeter_mean',data)
data = detect_outliers('area_mean',data)

# axis 1 to drop columns
# inplace=True argument means that the operation will be performed directly on the original data frame,
#  rather tahn creating a new data frame
data.drop('id', axis = 1, inplace = True)
data.drop('Unnamed: 32', axis=1, inplace=True)


x = data.drop('diagnosis', axis = 1)

y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#If using neural networks with libraries like PyTorch, you may need to use a custom normalization approach
#  (e.g., torch.nn.BatchNorm or manual normalization).


'''
param_grid = {
    'C':np.logspace(-20,0,20),
    'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'penalty':['l2','l1']
}

log_model=grid_search_run(param_grid,LogisticRegression())
y_pred=log_model.predict(X_test)
y_pred_train=log_model.predict(X_train)

# Print the best hyperparameters
print('Best Hyperparameters:', log_model.best_params_)

print('Logistic Regression')
print('Training Accuracy: ',accuracy_score(y_train,y_pred_train))
print('Testing Accuracy: ',accuracy_score(y_test,y_pred))
print('Precision: ',precision_score(y_test,y_pred))
print('Recall: ',recall_score(y_test,y_pred))
print('ROC_AUC: ',roc_auc_score(y_test,y_pred))
''' 


def getTestSetX():
    return X_test

def getTestSetY():
    return y_test

def getTrainSetX():
    return X_train

def getTrainSetY():
    return y_train