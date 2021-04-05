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


train_eng = pd.read_csv('./data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

data_dk_stories = pd.read_csv('./data/elastic_output/data_dk_stories.csv', index_col = 0)

egemaps_not_us_stories = pd.read_csv('./data/elastic_output/egemaps_not_us_stories.csv', index_col = 0)

data_dk_stories.head()
egemaps_not_us_stories.head()






####################################################### Training and validation #######################################################
# Define function for training/validation
def ml_validation(train, feature_lists, kernel = "linear", save = False):
    """
    Arguments:
    train           -   takes a df with diagnosis + all features
    feature_lists   -   takes a list of feature lists
    kernel          -   specify kernel, "rbf" or "linear"
    save            -   takes a Boolean, and if = True, it saves all confusion matrices + classifications reports separately, 
                        and also saves a df of predicted diagnosis + true diagnosis

    Output:
    A list of 3 elements
    element 1   -   a list of all classification reports
    element 2   -   a list of all confusion matrices
    element 3   -   a list, containing a df for each split, containing all predictions + true diagnosis
    """
    # Empty stuff for appending results
    classif_reports = ["", "", "", "", ""]
    conf_mtxs = ["", "", "", "", ""]
    model_predictions = ["", "", "", "", ""]

    # Model specifications
    if kernel == "rbf":
        model = SVC(kernel = 'rbf', class_weight = 'balanced')
    elif kernel == "linear":
        model = SVC(kernel = 'linear', class_weight = 'balanced')
    else:
        print("error \n Specify kernel: \"linear\" or \"rbf\"")

    # Creating a list of numbers from 1 to number of feature lists
    set_indices = list(range(1,len(feature_lists)+1))

    # Loop that does ML
    for n in set_indices:

        # For feature set 1 model, subset training data to only include fold 2,3,4,5
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        test = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'test' up into predictor variables (x) and what should be predicted (y)
        x_train = train_subset.loc[ : , feature_lists[n-1]]
        y_train = train_subset.loc[ : , 'Diagnosis'] # * Consider adding "ID" or "Gender", to be able to see differences in performance across genders?
        x_test = test.loc[ : , feature_lists[n-1]]
        y_test = test.loc[ : , 'Diagnosis']

        # Fitting the object "model" (which is the model) and predicting the test set
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        # Getting the performance
        classif_report = pd.DataFrame(classification_report(y_test, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

        # Loading the performance into the empty lists
        classif_reports[n-1] = classif_report
        conf_mtxs[n-1] = conf_matrix

        # Appending true and predicted diagnosis into the empty "model_predictions" dataframe
        true_diag_col_name = "".join(["fold_", str(n), "_true_diag"])
        pred_diag_col_name = "".join(["fold_", str(n), "_pred_diag"])
        model_prediction = pd.DataFrame()
        model_prediction[true_diag_col_name] = y_test
        model_prediction[pred_diag_col_name] = predictions
        model_predictions[n-1] = model_prediction

        # If save == True: save conf_matrix and classif_report
        if save == True:
            # Save names
            save_name_conf_matrix = os.path.join(".", "predictions", "validation", f"{kernel}_fold{str(n)}_conf_matrix.csv") 
            save_name_classif_report = os.path.join(".", "predictions", "validation", f"{kernel}_fold{str(n)}_classif_report.csv") 
                
            # Saving
            conf_matrix.to_csv(save_name_conf_matrix, sep=',', index = True)
            classif_report.to_csv(save_name_classif_report, sep=',', index = True)

    if save == True:
        # Saving the model predictions
        for i in range(1,6,1):
            pd.DataFrame(model_predictions[i-1]).to_csv(f"./predictions/validation/{kernel}_model_predictions_{i}.csv", sep=',', index = True)
            
    return(classif_reports, conf_mtxs, model_predictions)

# Define training/validation set
train_eng = pd.read_csv('./data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

# Define feature lists
feature_lists =  [train_eng.columns[3:], train_eng.columns[4:6], train_eng.columns[6:9], train_eng.columns[9:13], train_eng.columns[13:18]]# ** This needs changing, of coursE!!!!!!!!!!!! **

# Running the ml-validation
output_validation = ml_validation(train = train_eng, feature_lists = feature_lists, kernel = "rbf", save = True)

# Looking at output
output_validation[0][0] # Classification reports for submodels
output_validation[1][0] # Confusion matrices for submodels
output_validation[2][0] # Diagnosis predictions vs true diagnosis


####################################################### Training and testing #######################################################
# Define function for training/testing
def ml_test(train, test, feature_lists, kernel = "linear", save = False):
    """
    Arguments:
    train           -   takes a df with diagnosis + all features
    test            -   takes a df with diagnosis + all features
    feature_lists   -   takes a list of feature lists
    kernel          -   specify kernel, "rbf" or "linear"
    save            -   takes a Boolean, and if = True, it saves all confusion matrices + classifications reports separately, 
                        and also saves a df of predicted diagnosis + true diagnosis

    Output:
    A list of 3 elements
    element 1   -   a list of all classification reports
    element 2   -   a list of all confusion matrices
    element 3   -   a df, containing all predictions + true diagnosis
    element 4   -   a classification report of ensemble model
    element 5   -   a confusion report of ensemble model
    """
    # Empty stuff for appending results
    classif_reports = ["", "", "", "", ""]
    conf_mtxs = ["", "", "", "", ""]
    model_predictions = pd.DataFrame()

    # Model specifications
    if kernel == "rbf":
        model = SVC(kernel = 'rbf', class_weight = 'balanced')
    elif kernel == "linear":
        model = SVC(kernel = 'linear', class_weight = 'balanced')
    else:
        print("error \n Specify kernel: \"linear\" or \"rbf\"")

    # Creating a list of numbers from 1 to number of feature lists
    set_indices = list(range(1,len(feature_lists)+1))

    # Loop that does ML
    for n in set_indices:

        # For feature set 1 model, subset training data to only include fold 2,3,4,5
        train_subset = train.loc[train['.folds'] != n]

        # Dividing 'train' and 'test' up into predictor variables (x) and what should be predicted (y)
        x_train = train_subset.loc[ : , feature_lists[n-1]]
        y_train = train_subset.loc[ : , 'Diagnosis'] # * Consider adding "ID" or "Gender", to be able to see differences in performance across genders?
        x_test = test.loc[ : , feature_lists[n-1]]
        y_test = test.loc[ : , 'Diagnosis']

        # Fitting the object "model" (which is the model) and predicting the test set
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        # Getting the performance
        classif_report = pd.DataFrame(classification_report(y_test, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

        # Loading the performance into the empty lists
        classif_reports[n-1] = classif_report
        conf_mtxs[n-1] = conf_matrix

        # Appending true and predicted diagnosis into the empty "model_predictions" dataframe
        true_diag_col_name = "".join(["fold_", str(n), "_true_diag"])
        pred_diag_col_name = "".join(["fold_", str(n), "_pred_diag"])
        model_predictions[true_diag_col_name] = y_test
        model_predictions[pred_diag_col_name] = predictions

        # If save == True: save conf_matrix and classif_report
        if save == True:
            # Save names
            save_name_conf_matrix = os.path.join(".", "predictions", "test", f"{kernel}_fold{str(n)}_conf_matrix.csv") 
            save_name_classif_report = os.path.join(".", "predictions", "test", f"{kernel}_fold{str(n)}_classif_report.csv") 

            # Saving
            conf_matrix.to_csv(save_name_conf_matrix, sep=',', index = True)
            classif_report.to_csv(save_name_classif_report, sep=',', index = True)
    
    # Getting ensemble predictions into the model_predictions dataframe
    prediction_columns = ["fold_1_pred_diag", "fold_2_pred_diag", "fold_3_pred_diag", "fold_4_pred_diag", "fold_5_pred_diag"]
    ensemble_predictions = model_predictions[prediction_columns].mode(axis = 1)
    model_predictions["ensemble_predictions"] = ensemble_predictions

    # Performance for ensemble model
    ensemble_classif_report = pd.DataFrame(classification_report(y_test, ensemble_predictions, output_dict = True))
    ensemble_conf_matrix = pd.DataFrame(confusion_matrix(y_test, ensemble_predictions))

    if save == True:
        # Saving the model predictions
        pd.DataFrame(model_predictions).to_csv(f"./predictions/test/{kernel}_model_predictions.csv", sep=',', index = True)

        # Saving ensemble performance
        ensemble_classif_report.to_csv(f"./predictions/test/{kernel}_ensemble_classif_report.csv", sep=',', index = True)
        ensemble_conf_matrix.to_csv(f"./predictions/test/{kernel}_ensemble_conf_matrix.csv", sep=',', index = True)

    return(classif_reports, conf_mtxs, model_predictions, ensemble_classif_report, ensemble_conf_matrix)

# Define training set
train_eng = pd.read_csv('./data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

# Define testing set
test_eng_str = pd.read_csv('./data/lasso_gemaps_dk_testfold_1.csv', index_col = 0)

# Define feature lists
feature_lists =  [train_eng.columns[3:], train_eng.columns[4:6], train_eng.columns[6:9], train_eng.columns[9:13], train_eng.columns[13:18]]# ** This needs changing, of coursE!!!!!!!!!!!! **

# Running the ML
output = ml_test(train = train_eng, test = test_eng_str, feature_lists = feature_lists, kernel = "rbf", save = True)

# Looking at output
output[0][0] # Classification reports for individual "submodels"
output[1][0] # Confusion matrices for individual "submodels"
output[2] # All submodel predictions + ensemble predictions
output[3] # Ensemble_classif_report
output[4] # Ensemble_conf_matrix









