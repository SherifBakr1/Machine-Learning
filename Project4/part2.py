import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
def readData(file):
    df = pd.read_csv(file)
    return df

def preProcess(df):
    le = preprocessing.LabelEncoder()
    df['Credit'] = le.fit_transform(df['Credit']) # Changes A1, A2, etc. to encoded labels (0, 1, etc.)
    return df 

def evaluateClassifier(df, clf):
    features= df[Features].values
    labels = df[Label].values
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    auroc_scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test= features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.fit(X_train, y_train)
        y_pred= clf.predict_proba(X_test)[:,1]
        auroc = roc_auc_score(y_test, y_pred)
        auroc_scores.append(auroc)
    
    return auroc_scores

def runClassifiers(df):
    classifiers = {
        "Logistic Regression": LogisticRegression(), 
        "Naive Bayes": GaussianNB(), 
        "SVM": SVC(probability=True), 
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(), 
        "K-Nearest Neighbors": KNeighborsClassifier(), 
        "Gradient Boosting": GradientBoostingClassifier()
    }
    results = {'Classifier': [], 'AUROC per Fold': [], 'Avg AUROC': [], 'Std Deviation': []}

    for clf_name, clf in classifiers.items():
        auroc_scores= evaluateClassifier(df, clf)
        avg_auroc= round(np.mean(auroc_scores), 4)
        std_auroc= round(np.std(auroc_scores), 4)   
        results['Classifier'].append(clf_name)
        results['AUROC per Fold'].append(auroc_scores)
        results['Avg AUROC'].append(avg_auroc)
        results['Std Deviation'].append(std_auroc)

        if clf_name == "Random Forest":
            from sklearn.model_selection import train_test_split
            X = df[Features].values
            y = df[Label].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            #create a new classifier object
            new_clf = RandomForestClassifier()
            new_clf.fit(X_train, y_train)
            y_pred = new_clf.predict_proba(X_test)[:, 1]

            results_df = pd.DataFrame()
            results_df["Prediction"] = y_pred  
            results_df[Features + [Label]] = df  
            results_df.to_csv('bestModel.output', index=False)  # Save

            results_df= pd.read_csv('bestModel.output')
            
            threshold= 0.5
            y_true= y_test
            y_pred= (results_df['Prediction']>threshold).astype(int)

            conf_matrix= confusion_matrix(y_true, y_pred)
            print("Confusion Matrix: ")
            print(conf_matrix)

            accuracy= accuracy_score(y_true, y_pred)
            print("Accuracy: ", accuracy)

            recall= recall_score(y_true, y_pred)
            print("Recall: ", recall)

            auroc= roc_auc_score(y_true, y_pred)
            print("AUROC: ", auroc)

    results_df = pd.DataFrame(results)
    print(results_df)


Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
    clf.fit(df[Features].values, df[Label].values)
    return clf


if __name__ == "__main__":
    df = readData("credit_train.csv")
    df = preProcess(df)
    svc_param_grid= {
        'C': [0.1,1,10],
        'gamma': [0.1,1,10, 'scale', 'auto']
    }
    svc_tuned= GridSearchCV(SVC(probability=True), svc_param_grid, cv=10, scoring= 'roc_auc')
    svc_tuned.fit(df[Features].values, df[Label].values)
    print("Best SVC Hyperparameters: ", svc_tuned.best_params_)
    print("Best SVC AUROC:", round(svc_tuned.best_score_, 4))

    rf_param_grid = {
        'max_depth': [10, 20, 30, None], 
        'n_estimators': [100,200,300, 400]
    }
    rf_tuned= GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=10, scoring= 'roc_auc')
    rf_tuned.fit(df[Features].values, df[Label].values)
    print("Best Random Forest Hyperparameters: ", rf_tuned.best_params_)
    print("Best Random Forest AUROC: ", round(rf_tuned.best_score_, 4))

    #Save best model object
    best_randomforest_model= rf_tuned.best_estimator_
    best_randomforest_model_trained = trainOnAllData(df, best_randomforest_model)
    saveBestModel(best_randomforest_model_trained)

    runClassifiers(df)



