import pandas as pd
import numpy as np
import os
import scipy.stats
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold

########## TRAIN ##########
train_eng = pd.read_csv('./data/test_data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

########## TEST ##########
test_eng_str = pd.read_csv('./data/test_data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

########## FEATURE LISTS ##########
feature_lists =  [train_eng.columns[3:], train_eng.columns[4:6], train_eng.columns[6:9], train_eng.columns[9:13], train_eng.columns[13:18]]# ** This needs changing, of coursE!!!!!!!!!!!! **

########## DEFINING THE FUNCTION FOR TRAINING AND TESTING ##########
def ml(train, test, feature_lists, kernel = "linear", save = False):
    # Empty stuff for appending results
    classif_reports = ["", "", "", "", ""]
    conf_mtxs = ["", "", "", "", ""]
    model_predictions = pd.DataFrame()

    # Model specifications
    if kernel == "rbf":
        model = SVC(kernel = 'rbf', class_weight = 'balanced')
    elif kernel == "linear":
        model = SVC(kernel = 'linear', class_weight = 'balanced')

    # Creating a list of numbers from 1 to number of feature lists
    set_indices = list(range(1,len(feature_lists)+1))

    # Loop that does ML
    for n in set_indices:

        # Dividing 'train' and 'test' up into predictor variables (x) and what should be predicted (y)
        x_train = train.loc[ : , feature_lists[n-1]]
        y_train = train.loc[ : , 'Diagnosis'] # * Consider adding "ID" or "Gender", to be able to see differences in performance across genders?
        x_test = test.loc[ : , feature_lists[n-1]]
        y_test = test.loc[ : , 'Diagnosis']

        # Fitting the object "model" (which is the model) and predicting the test set
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        # Getting the performance
        classif_report = classification_report(y_test, predictions, output_dict = True)
        conf_matrix = confusion_matrix(y_test, predictions)

        # Loading the performance into the empty lists
        classif_reports[n-1] = classif_report
        conf_mtxs[n-1] = conf_matrix

        # Appending true and predicted diagnosis into the empty "model_predictions" dataframe
        true_diag_col_name = "".join(["fold_", str(n), "_pred_diag"])
        pred_diag_col_name = "".join(["fold_", str(n), "_true_diag"])
        model_predictions[true_diag_col_name] = y_test
        model_predictions[pred_diag_col_name] = predictions

        # If save = True: save conf_matrix and classif_report
        if save == True:
            # Save names
            save_name_conf_matrix = os.path.join(".", "predictions", f"fold{str(n)}_conf_matrix.csv") 
            save_name_classif_report = os.path.join(".", "predictions", f"fold{str(n)}_classif_report.csv") 
            
            # Saving
            pd.DataFrame(conf_matrix).to_csv(save_name_conf_matrix, sep=',', index = True)
            pd.DataFrame(classif_report).to_csv(save_name_classif_report, sep=',', index = True)

    if save == True:
        # Saving the model predictions
        pd.DataFrame(model_predictions).to_csv("./predictions/model_predictions.csv", sep=',', index = True)

    
    return(classif_reports, conf_mtxs, model_predictions)

# train - takes a df with diagnosis + all features
# test - takes a df with diagnosis + all feature
# feature_lists - takes a list of feature lists
# kernel - specify kernel, "rbf" or "linear"
# save - takes a Boolean, and if = True, it saves all confusion matrices + classifications reports separately, and also saves a df of predicted diagnosis + true diagnosis

# the function returns a list of 3 elements
# element 1 - a list of all classification reports
# element 2 - a list of all confusion matrices
# element 3 - a df, containing all predictions + true diagnosis

########## Trying it ##########
output = ml(train = train_eng, test = test_eng_str, feature_lists = feature_lists, kernel = "linear", save = True)



