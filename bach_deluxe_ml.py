import scipy.stats
import pandas as pd
import numpy as np
import os, random, joblib, statistics
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
random.seed(1213870)

# Load training + validation sets
data_dk_stories = pd.read_csv('./data/elastic_output/data_egemaps_dk_stories.csv', index_col = 0)
data_dk_triangles = pd.read_csv('./data/elastic_output/data_egemaps_dk_triangles.csv', index_col = 0)
data_us_stories = pd.read_csv('./data/elastic_output/data_egemaps_us_stories.csv', index_col = 0)

# Load test sets
model_dk_stories_test_on_dk_stories = pd.read_csv("./data/elastic_output/egemaps_model_dk_stories_test_on_dk_stories.csv", index_col = 0)
model_dk_stories_test_on_not_dk = pd.read_csv("./data/elastic_output/egemaps_model_dk_stories_test_on_not_dk.csv", index_col = 0)
model_dk_stories_test_on_not_stories = pd.read_csv("./data/elastic_output/egemaps_model_dk_stories_test_on_not_stories.csv", index_col = 0)
model_dk_triangles_test_on_dk_triangles = pd.read_csv("./data/elastic_output/egemaps_model_dk_triangles_test_on_dk_triangles.csv", index_col = 0)
model_dk_triangles_test_on_not_dk = pd.read_csv("./data/elastic_output/egemaps_model_dk_triangles_test_on_not_dk.csv", index_col = 0)
model_dk_triangles_test_on_not_triangles = pd.read_csv("./data/elastic_output/egemaps_model_dk_triangles_test_on_not_triangles.csv", index_col = 0)
model_us_stories_test_on_not_us = pd.read_csv("./data/elastic_output/egemaps_model_us_stories_test_on_not_us.csv", index_col = 0)
model_us_stories_test_on_us_stories = pd.read_csv("./data/elastic_output/egemaps_model_us_stories_test_on_us_stories.csv", index_col = 0)

# Load feature sets for dk_stories (and convert to list of lists)
features_dk_stories_df = pd.read_csv('./data/feature_lists/features_egemaps_dk_stories.csv', index_col = 0)
features_dk_stories = []
for i in range(1,6,1):
    single_set = features_dk_stories_df.loc[features_dk_stories_df['fold'] == i]["features"].tolist()
    features_dk_stories.append(single_set)

# Load feature sets for dk_triangles (and convert to list of lists)
features_dk_triangles_df = pd.read_csv('./data/feature_lists/features_egemaps_dk_triangles.csv', index_col = 0)
features_dk_triangles = []
for i in range(1,6,1):
    single_set = features_dk_triangles_df.loc[features_dk_triangles_df['fold'] == i]["features"].tolist()
    features_dk_triangles.append(single_set)

# Load feature sets for us_stories (and convert to list of lists)
features_us_stories_df = pd.read_csv('./data/feature_lists/features_egemaps_us_stories.csv', index_col = 0)
features_us_stories = []
for i in range(1,6,1):
    single_set = features_us_stories_df.loc[features_us_stories_df['fold'] == i]["features"].tolist()
    features_us_stories.append(single_set)

# Defining ml_train
def ml_train(train, train_name, feature_lists, kernel, save):
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

        # Fit model to training data and save the model
        model = model.fit(trainX, trainY)
        if not os.path.exists(f'./data/models/{train_name}/'):
            os.makedirs(f'./data/models/{train_name}/')
        joblib.dump(model, f'./data/models/{train_name}/{kernel}_{n}.pkl')

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
    return validation_classification_reports, validation_confusion_matrices, validation_model_predictions

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
        predictions = joblib.load(f'./data/models/{train_name}/{kernel}_{n}.pkl').predict(testX)

        # Retrieving performance measures
        classif_report = pd.DataFrame(classification_report(testY, predictions, output_dict = True))
        conf_matrix = pd.DataFrame(confusion_matrix(testY, predictions))

        # Loading the performance into the empty lists
        classification_reports.append(classif_report)
        confusion_matrices.append(conf_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe    
        model_predictions[f"model_{str(n)}_predicted_diagnosis"] = predictions

    # Getting majority decision of the 5 models and appending it to the df "model_predictions" 
    ensemble_predictions = model_predictions[["model_1_predicted_diagnosis", "model_2_predicted_diagnosis", "model_3_predicted_diagnosis", "model_4_predicted_diagnosis", "model_5_predicted_diagnosis"]].mode(axis = 1)
    
    # Adding Gender to be able to see performance of the ensemble model across genders
    ensemble_predictions["Gender"] = test["Gender"]

    # Appending new column with ensemble predictions
    model_predictions["ensemble_predictions"] = ensemble_predictions.iloc[:,0]
    
    # Appending new column with sex of participants
    model_predictions["Gender"] = ensemble_predictions.iloc[:,1]

    # Getting the classification report + confusion matrix for the ensemble model. Both sexes.
    ensemble_classification_report = pd.DataFrame(classification_report(testY, ensemble_predictions.iloc[:,0], output_dict = True))
    ensemble_confusion_matrix = pd.DataFrame(confusion_matrix(testY, ensemble_predictions.iloc[:,0]))

    # Getting the classification report + confusion matrix for the ensemble model. Female
    ensemble_classification_report_female = pd.DataFrame(classification_report(test[test["Gender"] == "Female"]["Diagnosis"], ensemble_predictions[ensemble_predictions["Gender"] == "Female"].iloc[:,0], output_dict = True))
    ensemble_confusion_matrix_female = pd.DataFrame(confusion_matrix(test[test["Gender"] == "Female"]["Diagnosis"], ensemble_predictions[ensemble_predictions["Gender"] == "Female"].iloc[:,0]))

    # Getting the classification report + confusion matrix for the ensemble model. Male
    ensemble_classification_report_male = pd.DataFrame(classification_report(test[test["Gender"] == "Male"]["Diagnosis"], ensemble_predictions[ensemble_predictions["Gender"] == "Male"].iloc[:,0], output_dict = True))
    ensemble_confusion_matrix_male = pd.DataFrame(confusion_matrix(test[test["Gender"] == "Male"]["Diagnosis"], ensemble_predictions[ensemble_predictions["Gender"] == "Male"].iloc[:,0]))

    # Saving output
    if save == True:
        # Save the predictions
        model_predictions.to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_model_predictions.csv", sep=',', index = True)

        # Save the ensemble classification report + confusion matrix - both sexes and males and females respectively
        pd.DataFrame(ensemble_classification_report).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble.csv", sep=',', index = True)
        pd.DataFrame(ensemble_confusion_matrix).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble.csv", sep=',', index = True)
        pd.DataFrame(ensemble_classification_report_female).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble_female.csv", sep=',', index = True)
        pd.DataFrame(ensemble_confusion_matrix_female).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble_female.csv", sep=',', index = True)
        pd.DataFrame(ensemble_classification_report_male).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble_male.csv", sep=',', index = True)
        pd.DataFrame(ensemble_confusion_matrix_male).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble_male.csv", sep=',', index = True)

        # Save the individual model classification reports + confusion matrices
        for n in index_list:
            # Go through each index in the list of classification reports  - save them
            pd.DataFrame(classification_reports[n-1]).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_{n}.csv", sep=',', index = True)

            # Go through each index in the list of confusion matrices  - save them
            pd.DataFrame(confusion_matrices[n-1]).to_csv(f"./predictions/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_{n}.csv", sep=',', index = True)

    return classification_reports, confusion_matrices, model_predictions, ensemble_classification_report, ensemble_confusion_matrix, ensemble_classification_report_female, ensemble_confusion_matrix_female, ensemble_classification_report_male, ensemble_confusion_matrix_male

# Training models (and validating models)
output = ml_train(train = data_dk_stories, train_name = "dk_stories", feature_lists = features_dk_stories, kernel = "rbf", save = False)
output = ml_train(train = data_dk_triangles, train_name = "dk_triangles", feature_lists = features_dk_triangles, kernel = "rbf", save = False)
output = ml_train(train = data_us_stories, train_name = "us_stories", feature_lists = features_us_stories, kernel = "rbf", save = False)

# Testing using dk_stories model
output = ml_test(train_name = "dk_stories", test_name = "dk_stories", kernel = "rbf", save = False, test = model_dk_stories_test_on_dk_stories, feature_lists = features_dk_stories)
#output = ml_test(train_name = "dk_stories", test_name = "us_stories", kernel = "rbf", save = False, test = model_dk_stories_test_on_not_dk, feature_lists = features_dk_stories)
output = ml_test(train_name = "dk_stories", test_name = "dk_triangles", kernel = "rbf", save = False, test = model_dk_stories_test_on_not_stories, feature_lists = features_dk_stories)

# Testing using dk_triangles model
output = ml_test(train_name = "dk_triangles", test_name = "dk_stories", kernel = "rbf", save = False, test = model_dk_triangles_test_on_not_triangles, feature_lists = features_dk_triangles)
#output = ml_test(train_name = "dk_triangles", test_name = "us_stories", kernel = "rbf", save = False, test = model_dk_triangles_test_on_not_dk, feature_lists = features_dk_triangles)
output = ml_test(train_name = "dk_triangles", test_name = "dk_triangles", kernel = "rbf", save = False, test = model_dk_triangles_test_on_dk_triangles, feature_lists = features_dk_triangles)

# Testing using us_stories model
#output = ml_test(train_name = "us_stories", test_name = "dk_stories", kernel = "rbf", save = False, test = model_us_stories_test_on_not_us, feature_lists = features_us_stories)
output = ml_test(train_name = "us_stories", test_name = "us_stories", kernel = "rbf", save = False, test = model_us_stories_test_on_us_stories, feature_lists = features_us_stories)


### Looking individually at all the most relevant performances (won't be in the final script on our github) ###

# Training models
output = ml_train(train = data_dk_stories, train_name = "dk_stories", feature_lists = features_dk_stories, kernel = "rbf", save = False)
classification_reports, confusion_matrices, model_predictions = output
macro_avg_f1 = []
for i in range(0,5,1):
    macro_avg_f1.append(classification_reports[i].iloc[2,4])
statistics.mean(macro_avg_f1) # Mean of validation macro average f1 scores for both sexes
# .67

output = ml_train(train = data_dk_triangles, train_name = "dk_triangles", feature_lists = features_dk_triangles, kernel = "rbf", save = False)
classification_reports, confusion_matrices, model_predictions = output
macro_avg_f1 = []
for i in range(0,5,1):
    macro_avg_f1.append(classification_reports[i].iloc[2,4]) 
statistics.mean(macro_avg_f1) # Mean of validation macro average f1 scores for both sexes
# .7

output = ml_train(train = data_us_stories, train_name = "us_stories", feature_lists = features_us_stories, kernel = "rbf", save = False)
classification_reports, l, l = output
macro_avg_f1 = []
for i in range(0,5,1):
    macro_avg_f1.append(classification_reports[i].iloc[2,4])
statistics.mean(macro_avg_f1) # Mean of validation macro average f1 scores for both sexes
# .65


# Testing using dk_stories
output = ml_test(train_name = "dk_stories", test_name = "dk_stories", kernel = "rbf", save = False, test = model_dk_stories_test_on_dk_stories, feature_lists = features_dk_stories)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score for both sexes
# 57
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.77
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.41

output = ml_test(train_name = "dk_stories", test_name = "us_stories", kernel = "rbf", save = False, test = model_dk_stories_test_on_not_dk, feature_lists = features_dk_stories)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score for both sexes
# 43
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.17
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.45

output = ml_test(train_name = "dk_stories", test_name = "dk_triangles", kernel = "rbf", save = False, test = model_dk_stories_test_on_not_stories, feature_lists = features_dk_stories)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score for both sexes
# 67
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.68
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.67


# Testing using dk_triangles
output = ml_test(train_name = "dk_triangles", test_name = "dk_stories", kernel = "rbf", save = False, test = model_dk_triangles_test_on_not_triangles, feature_lists = features_dk_triangles)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score both sexes
# 68
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.74
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.65

output = ml_test(train_name = "dk_triangles", test_name = "us_stories", kernel = "rbf", save = False, test = model_dk_triangles_test_on_not_dk, feature_lists = features_dk_triangles)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score both sexes
# 46
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.30
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.47

output = ml_test(train_name = "dk_triangles", test_name = "dk_triangles", kernel = "rbf", save = False, test = model_dk_triangles_test_on_dk_triangles, feature_lists = features_dk_triangles)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score both sexes
# 69
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.75
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.62


# Testing using us_stories
output = ml_test(train_name = "us_stories", test_name = "dk_stories", kernel = "rbf", save = False, test = model_us_stories_test_on_not_us, feature_lists = features_us_stories)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score both sexes
# 56
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.68
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.53

output = ml_test(train_name = "us_stories", test_name = "us_stories", kernel = "rbf", save = False, test = model_us_stories_test_on_us_stories, feature_lists = features_us_stories)
l, l, l, ensemble_classification_report, l, ensemble_classification_report_female, l, ensemble_classification_report_male, l, = output
ensemble_classification_report.iloc[2,4] # Macro average f1 score both sexes
# 38
ensemble_classification_report_female.iloc[2,4] # Macro average f1 score for females
# 0.14
ensemble_classification_report_male.iloc[2,4] # Macro average f1 score for males
# 0.44
