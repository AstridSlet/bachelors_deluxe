import pandas as pd
import numpy as np
import os, random, joblib
import scipy.stats
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
random.seed(1213870)

############################################################ Loading data ##########################################################
# Load training + validation sets
data_dk_stories = pd.read_csv('./data/elastic_output/data_dk_stories.csv', index_col = 0)
data_dk_triangles = pd.read_csv('./data/elastic_output/data_dk_triangles.csv', index_col = 0)
data_us_stories = pd.read_csv('./data/elastic_output/data_us_stories.csv', index_col = 0)

# Load test sets
model_dk_stories_test_on_dk_stories = pd.read_csv("./data/elastic_output/model_dk_stories_test_on_dk_stories.csv", index_col = 0)
model_dk_stories_test_on_not_dk = pd.read_csv("./data/elastic_output/model_dk_stories_test_on_not_dk.csv", index_col = 0)
model_dk_stories_test_on_not_stories = pd.read_csv("./data/elastic_output/model_dk_stories_test_on_not_stories.csv", index_col = 0)
model_dk_triangles_test_on_dk_triangles = pd.read_csv("./data/elastic_output/model_dk_triangles_test_on_dk_triangles.csv", index_col = 0)
model_dk_triangles_test_on_not_dk = pd.read_csv("./data/elastic_output/model_dk_triangles_test_on_not_dk.csv", index_col = 0)
model_dk_triangles_test_on_not_triangles = pd.read_csv("./data/elastic_output/model_dk_triangles_test_on_not_triangles.csv", index_col = 0)
model_us_stories_test_on_not_us = pd.read_csv("./data/elastic_output/model_us_stories_test_on_not_us.csv", index_col = 0)
model_us_stories_test_on_us_stories = pd.read_csv("./data/elastic_output/model_us_stories_test_on_us_stories.csv", index_col = 0)

# Load feature sets for dk_stories (and convert to list of lists)
features_dk_stories_df = pd.read_csv('./data/feature_lists/features_dk_stories.csv', index_col = 0)
features_dk_stories = []
for i in range(1,6,1):
    single_set = features_dk_stories_df.loc[features_dk_stories_df['fold'] == i]["features"].tolist()
    features_dk_stories.append(single_set)

# Load feature sets for dk_triangles (and convert to list of lists)
features_dk_triangles_df = pd.read_csv('./data/feature_lists/features_dk_triangles.csv', index_col = 0)
features_dk_triangles = []
for i in range(1,6,1):
    single_set = features_dk_triangles_df.loc[features_dk_triangles_df['fold'] == i]["features"].tolist()
    features_dk_triangles.append(single_set)

# Load feature sets for us_stories (and convert to list of lists)
features_us_stories_df = pd.read_csv('./data/feature_lists/features_us_stories.csv', index_col = 0)
features_us_stories = []
for i in range(1,6,1):
    single_set = features_us_stories_df.loc[features_us_stories_df['fold'] == i]["features"].tolist()
    features_us_stories.append(single_set)


# Defining ml_train
def ml_train(train, train_name, feature_lists, kernel, save):
    # Empty list which is to contain the models
    #models = []

    # Empty lists for appending
    validation_classification_reports = []
    validation_confusion_matrices = []
    validation_model_predictions = []

    # Model specifications
    if kernel == "rbf":
        model = SVC(kernel='rbf', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' , 
    elif kernel == "linear":
        model = SVC(kernel='linear', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' ,
    else:
        print("error \n Specify kernel: \"linear\" or \"rbf\"")

    # Creating a list of numbers from 1 to number of feature lists
    index_list = list(range(1,len(feature_lists)+1))

    # Loop that trains and validates
    for n in index_list:
        # For feature set 1 model, subset training data to only include fold 2,3,4,5. Etc.
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        validation = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
        trainX = train_subset.loc[ : , feature_lists[n-1]]
        trainY = train_subset.loc[ : , 'Diagnosis'] # * Consider adding "ID" or "Gender", to be able to see differences in performance across genders?
        validationX = validation.loc[ : , feature_lists[n-1]]
        validationY = validation.loc[ : , 'Diagnosis']

        # Fit model to training data and append to model list
        model = model.fit(trainX, trainY)
        #models.append(model)
        joblib.dump(model, f'./data/models/{train_name}/{n}.pkl')

        # Predict validation set with model
        validation_predictions = model.predict(validationX)

        # Retrieving performance measures
        validation_classification_report = pd.DataFrame(classification_report(validationY, validation_predictions, output_dict = True))
        validation_confusion_matrix = pd.DataFrame(confusion_matrix(validationY, validation_predictions))

        # Loading the performance into the empty lists
        validation_classification_reports.append(validation_classification_report)
        validation_confusion_matrices.append(validation_confusion_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions = pd.DataFrame({f"fold_{str(n)}_true_diagnosis": validationY, f"fold_{str(n)}_predicted_diagnosis": validation_predictions})
        validation_model_predictions.append(model_predictions)

    if save == True:
        for n in index_list:
            # Go through each index in the list of data frames with diagnosis predictions and true diagnosis - save them
            pd.DataFrame(validation_model_predictions[n-1]).to_csv(f"./predictions/validation/{train_name}/{train_name}_{kernel}_model_predictions_{n}.csv", sep=',', index = True)
            
            # Go through each index in the list of classification reports  - save them
            pd.DataFrame(validation_classification_reports[n-1]).to_csv(f"./predictions/validation/{train_name}/{train_name}_{kernel}_classification_report_{n}.csv", sep=',', index = True)

            # Go through each index in the list of confusion matrices  - save them
            pd.DataFrame(validation_confusion_matrices[n-1]).to_csv(f"./predictions/validation/{train_name}/{train_name}_{kernel}_confusion_matrix_{n}.csv", sep=',', index = True)
    
    # Return the kernel + the 4 lists
    return kernel, validation_classification_reports, validation_confusion_matrices, validation_model_predictions

# Defining ml_test
def ml_test(train_name, test_name, kernel, save, test, feature_lists):
    # Empty lists for appending
    classification_reports = []
    confusion_matrices = []
    model_predictions = pd.DataFrame({"true_diagnosis": test["Diagnosis"]})

    # Creating a list of numbers from 1 to number of feature lists
    index_list = list(range(1,len(feature_lists)+1))

    # Loop that trains and validates
    for n in index_list:
        
        # Divide up the test set into predictor variables (testX) and what should be predicted (testY)
        testX = test.loc[ : , feature_lists[n-1]]
        testY = test.loc[ : , 'Diagnosis']

        # Predict test set with saved model
        predictions = joblib.load(f'./data/models/{train_name}/{n}.pkl').predict(testX)

        # Retrieving performance measures
        classif_report = pd.DataFrame(classification_report(testY, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(testY, predictions))

        #print(f"{classif_report} \n new \n")

        # Loading the performance into the empty lists
        classification_reports.append(classif_report)
        confusion_matrices.append(conf_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions[f"model_{str(n)}_predicted_diagnosis"] = predictions

    # Getting majority decision of the 5 models and appending it to the df "model_predictions" 
    ensemble_predictions = model_predictions[["model_1_predicted_diagnosis", "model_2_predicted_diagnosis", "model_3_predicted_diagnosis", "model_4_predicted_diagnosis", "model_5_predicted_diagnosis"]].mode(axis = 1)
    model_predictions["ensemble_predictions"] = ensemble_predictions

    # Getting the classification report + confusion matrix for the ensemble model
    ensemble_classification_report = pd.DataFrame(classification_report(testY, ensemble_predictions, output_dict = True))
    ensemble_confusion_matrix = pd.DataFrame(confusion_matrix(testY, ensemble_predictions))

    # Saving output
    if save == True:
        # Save the predictions
        model_predictions.to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_model_predictions.csv", sep=',', index = True)

        # Save the ensemble classification report + confusion matrix
        pd.DataFrame(ensemble_classification_report).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble.csv", sep=',', index = True)
        pd.DataFrame(ensemble_confusion_matrix).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble.csv", sep=',', index = True)

        # Save the individual model classification reports + confusion matrices
        for n in index_list:
            # Go through each index in the list of classification reports  - save them
            pd.DataFrame(classification_reports[n-1]).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_{n}.csv", sep=',', index = True)

            # Go through each index in the list of confusion matrices  - save them
            pd.DataFrame(confusion_matrices[n-1]).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_{n}.csv", sep=',', index = True)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix

# Training models
ml_train(train = data_dk_stories, train_name = "dk_stories", feature_lists = features_dk_stories, kernel = "rbf", save = True)
ml_train(train = data_dk_triangles, train_name = "dk_triangles", feature_lists = features_dk_triangles, kernel = "rbf", save = True)
ml_train(train = data_us_stories, train_name = "us_stories", feature_lists = features_us_stories, kernel = "rbf", save = True)

# Testing using dk_stories
ml_test(train_name = "dk_stories", test_name = "dk_stories", kernel = "rbf", save = True, test = model_dk_stories_test_on_dk_stories, feature_lists = features_dk_stories)
ml_test(train_name = "dk_stories", test_name = "us_stories", kernel = "rbf", save = True, test = model_dk_stories_test_on_not_dk, feature_lists = features_dk_stories)
ml_test(train_name = "dk_stories", test_name = "dk_triangles", kernel = "rbf", save = True, test = model_dk_stories_test_on_not_stories, feature_lists = features_dk_stories)

# Testing using dk_triangles
ml_test(train_name = "dk_triangles", test_name = "dk_stories", kernel = "rbf", save = True, test = model_dk_triangles_test_on_not_triangles, feature_lists = features_dk_triangles)
ml_test(train_name = "dk_triangles", test_name = "us_stories", kernel = "rbf", save = True, test = model_dk_triangles_test_on_not_dk, feature_lists = features_dk_triangles)
ml_test(train_name = "dk_triangles", test_name = "dk_triangles", kernel = "rbf", save = True, test = model_dk_triangles_test_on_dk_triangles, feature_lists = features_dk_triangles)

# Testing using us_stories
ml_test(train_name = "us_stories", test_name = "dk_stories", kernel = "rbf", save = True, test = model_us_stories_test_on_not_us, feature_lists = features_us_stories)
ml_test(train_name = "us_stories", test_name = "us_stories", kernel = "rbf", save = True, test = model_us_stories_test_on_us_stories, feature_lists = features_us_stories)


























































































####################################################### Training and validation #######################################################
# Define function for training/validation
def ml_validation(train, train_name, feature_lists, kernel = "linear", save = False):
    """
    Arguments:
    train           -   takes a df with diagnosis + all features
    train_name      -   takes a str, which contains name of training set
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
    classif_reports = []
    conf_mtxs = []
    model_predictions = []

    # Model specifications
    if kernel == "rbf":
        model = SVC(kernel='rbf', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' , 
    elif kernel == "linear":
        model = SVC(kernel='linear', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' ,
    else:
        print("error \n Specify kernel: \"linear\" or \"rbf\"")

    # Creating a list of numbers from 1 to number of feature lists
    set_indices = list(range(1,len(feature_lists)+1))

    # Loop that does ML
    for n in set_indices:

        # For feature set 1 model, subset training data to only include fold 2,3,4,5
        train_subset = train.loc[train['.folds'] != n]

        # Defining validation set
        validation = train.loc[train['.folds'] == n]

        # Dividing 'train' and 'validation' up into predictor variables (x) and what should be predicted (y)
        x_train = train_subset.loc[ : , feature_lists[n-1]]
        y_train = train_subset.loc[ : , 'Diagnosis'] # * Consider adding "ID" or "Gender", to be able to see differences in performance across genders?
        x_validation = validation.loc[ : , feature_lists[n-1]]
        y_validation = validation.loc[ : , 'Diagnosis']

        # Fitting the object "model" (which is the model) and predicting the validation set
        model.fit(x_train, y_train)
        predictions = model.predict(x_validation)
        
        # Getting the performance
        classif_report = pd.DataFrame(classification_report(y_validation, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(y_validation, predictions))

        # Loading the performance into the empty lists
        classif_reports.append(classif_report)
        conf_mtxs.append(conf_matrix) 

        # Appending true and predicted diagnosis into the empty "model_predictions" dataframe
        true_diag_col_name = "".join(["fold_", str(n), "_true_diag"])
        pred_diag_col_name = "".join(["fold_", str(n), "_pred_diag"])
        model_prediction = pd.DataFrame()
        model_prediction[true_diag_col_name] = y_validation
        model_prediction[pred_diag_col_name] = predictions
        model_predictions.append(model_prediction)

        # If save == True: save conf_matrix and classif_report
        if save == True:
            # Save names
            save_name_conf_matrix = os.path.join(".", "predictions", "validation", f"{train_name}", f"{train_name}_{kernel}_fold{str(n)}_conf_matrix.csv") 
            save_name_classif_report = os.path.join(".", "predictions", "validation", f"{train_name}", f"{train_name}_{kernel}_fold{str(n)}_classif_report.csv") 
            
            # Saving
            conf_matrix.to_csv(save_name_conf_matrix, sep=',', index = True)
            classif_report.to_csv(save_name_classif_report, sep=',', index = True)

    if save == True:
        # Saving the model predictions
        for i in range(1,6,1):
            pd.DataFrame(model_predictions[i-1]).to_csv(f"./predictions/validation/{train_name}/{train_name}_{kernel}_model_predictions_{i}.csv", sep=',', index = True)
            
    return(classif_reports, conf_mtxs, model_predictions)

# Running the ml_validation
output_validation = ml_validation(train = data_dk_stories, train_name = "dk_stories", feature_lists = features_dk_stories, kernel = "linear", save = False)
output_validation = ml_validation(train = data_dk_triangles, train_name = "dk_triangles", feature_lists = features_dk_triangles, kernel = "rbf", save = False)
output_validation = ml_validation(train = data_us_stories, train_name = "us_stories", feature_lists = features_us_stories, kernel = "rbf", save = False)

####################################################### Training and testing #######################################################
# Define function for training/testing
def ml_test(train, train_name, test_name,  test, feature_lists, kernel = "linear", save = False):
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
        model = SVC(kernel='rbf', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' , 
    elif kernel == "linear":
        model = SVC(kernel='linear', class_weight = 'balanced') #default is gamma scaled, else use gamma='auto' ,         
    else:
        print("error \n Specify kernel: \"linear\" or \"rbf\"")

    # Creating a list of numbers from 1 to number of feature lists
    set_indices = list(range(1,len(feature_lists)+1))

    # Loop that does ML
    for n in set_indices:

        # For feature set 1 model, subset training data to only include fold 2,3,4,5 etc.
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
            save_name_conf_matrix = os.path.join(".", "predictions", "test", f"{train_name}", f"{train_name}_tested_on_{test_name}_{kernel}_fold{str(n)}_conf_matrix.csv") 
            save_name_classif_report = os.path.join(".", "predictions", "test", f"{train_name}", f"{train_name}_tested_on_{test_name}_{kernel}_fold{str(n)}_classif_report.csv") 

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
        pd.DataFrame(model_predictions).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_model_predictions.csv", sep=',', index = True)

        # Saving ensemble performance
        ensemble_classif_report.to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_ensemble_classif_report.csv", sep=',', index = True)
        ensemble_conf_matrix.to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_ensemble_conf_matrix.csv", sep=',', index = True)

    return(classif_reports, conf_mtxs, model_predictions, ensemble_classif_report, ensemble_conf_matrix)

# Running the ML_test
# Danish story model:
#output_test = ml_test(train = data_dk_stories, train_name = "dk_stories", test_name = "dk_stories", test = egemaps_not_us_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)
output_test = ml_test(train = data_dk_stories, train_name = "dk_stories", test_name = "dk_triangles", test = egemaps_dk_not_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)
output_test = ml_test(train = data_dk_stories, train_name = "dk_stories", test_name = "us_stories", test = egemaps_not_dk_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)

# Danish triangle model:
output_test = ml_test(train = data_dk_triangles, train_name = "dk_triangles", test_name = "dk_stories", test = egemaps_dk_not_triangles, feature_lists = features_dk_triangles, kernel = "rbf", save = True)
#output_test = ml_test(train = data_dk_triangles, train_name = "dk_triangles", test_name = "dk_triangles", test = egemaps_not_us_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)
#output_test = ml_test(train = data_dk_triangles, train_name = "dk_triangles", test_name = "us_stories", test = egemaps_not_dk_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)

# US story model:
output_test = ml_test(train = data_us_stories, train_name = "us_stories", test_name = "dk_stories", test = egemaps_not_us_stories, feature_lists = features_us_stories, kernel = "rbf", save = True)
#output_test = ml_test(train = data_us_stories, train_name = "us_stories", test_name = "dk_triangles", test = egemaps_not_us_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)
#output_test = ml_test(train = data_us_stories, train_name = "us_stories", test_name = "us_stories", test = egemaps_not_us_stories, feature_lists = features_dk_stories, kernel = "rbf", save = True)

