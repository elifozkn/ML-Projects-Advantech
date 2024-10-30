from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import xgboost as xgb
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from connect_to_database import get_training_data,write_to_sql

def hyperparameter_tuning_xgb(X,y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

    # Create a scorer based on F1-score
    f1_scorer = make_scorer(f1_score)

    # Perform stratified 5-fold cross-validation with GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=f1_scorer, cv=cv, verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model with the optimal hyperparameters
    best_xgb_model = grid_search.best_estimator_

    return best_xgb_model


def train_xgboost(df): 

    X = df.drop(['Label','target_encoded','Customer_Id','cust_country'],axis = 1)
    X['Customer_AOL'] = X['Customer_AOL'].astype(int)
    X['Customer_CP'] = X['Customer_CP'].astype(int)
    X['Customer_KA'] = X['Customer_KA'].astype(int)
    y = df['target_encoded']

    model = hyperparameter_tuning_xgb(X, y)
    # Initialize dictionaries to store metrics
    precision_dict = {'0': [], '1': []}
    recall_dict = {'0': [], '1': []}
    f1_dict = {'0': [], '1': []}
    support_dict = {'0': [], '1': []}
    accuracy_list = []

    # Perform stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classification_reports = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = pd.DataFrame(X).iloc[train_idx], pd.DataFrame(X).iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Collect metrics from the classification report
        precision_dict['0'].append(report['0']['precision'])
        precision_dict['1'].append(report['1']['precision'])
        recall_dict['0'].append(report['0']['recall'])
        recall_dict['1'].append(report['1']['recall'])
        f1_dict['0'].append(report['0']['f1-score'])
        f1_dict['1'].append(report['1']['f1-score'])
        support_dict['0'].append(report['0']['support'])
        support_dict['1'].append(report['1']['support'])
        accuracy_list.append(report['accuracy'])

    # Calculate and print the average classification report
    avg_classification_report = {
        '0': {
            'precision': np.mean(precision_dict['0']),
            'recall': np.mean(recall_dict['0']),
            'f1-score': np.mean(f1_dict['0']),
            'support': np.sum(support_dict['0'])
        },
        '1': {
            'precision': np.mean(precision_dict['1']),
            'recall': np.mean(recall_dict['1']),
            'f1-score': np.mean(f1_dict['1']),
            'support': np.sum(support_dict['1'])
        },
        'accuracy': np.mean(accuracy_list)
}

    # Save the trained model to a file
    model_filename = r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\src\models\xgboost.model'  # Replace with your desired file name
    model.save_model(model_filename)

    #print(f"Trained model saved as {model_filename}")

    return model_filename,y_pred,y_pred_prob


def hyperparameter_tuning_MLP(data):
    # Define the hyperparameter grid to search
    X = data.drop(['Label','target_encoded','Customer_Id','cust_country'],axis = 1)
    y = data['target_encoded']
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (64, 64, 32), (128, 64), (128, 64, 32)],
        'activation': ['relu', 'tanh','sigmoid'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    # Create the MLP model
    mlp = MLPClassifier(max_iter=1000, random_state=42)

    # Initialize the GridSearchCV
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', verbose=1)

    # Fit the grid search to your data
    grid_search.fit(X, y)

    # Print the best hyperparameters and their corresponding F1 score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best F1 Score: ", grid_search.best_score_)

    return grid_search.best_params_

def train_MLP(data):
    
    X = data.drop(['Label','target_encoded','Customer_Id','cust_country'],axis = 1)
    y = data['target_encoded']

    # Define StratifiedKFold for 5-fold cross-validation
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store evaluation results for each fold
    fold_accuracies = []
    fold_losses = []
    fold_f1_scores = []

    # Loop through each fold
    for train_indices, test_indices in stratified_kfold.split(X, y):
        # Split the data into training and testing sets for this fold
        X_train, X_test = pd.DataFrame(X).iloc[train_indices], pd.DataFrame(X).iloc[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Convert X_train and X_test DataFrames to NumPy arrays
        X_train = X_train.values  # Converts X_train DataFrame to a NumPy array
        X_test = X_test.values    # Converts X_test DataFrame to a NumPy array
        y_train = y_train.values 
        y_test = y_test.values
       
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        # Create an MLP model
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
        model.add(layers.Dense(64, activation='relu'))  # Hidden layer with 64 units and ReLU activation
        model.add(layers.Dense(32, activation='relu'))  # Another hidden layer with 32 units and ReLU activation
        model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #
        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate the model on the test data for this fold
        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Calculate and append F1 score for this fold
        f1 = f1_score(y_test, y_pred)
        fold_f1_scores.append(f1)

        # Print evaluation metrics for this fold
        print(f'Fold Test Loss: {loss:.4f}')
        print(f'Fold Test Accuracy: {accuracy:.4f}')
        print(f'Fold F1 Score: {f1:.4f}')

        # Append fold results to the lists
        fold_accuracies.append(accuracy)
        fold_losses.append(loss)

    # Calculate and print the mean and standard deviation of fold accuracies
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f'Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f} std dev)')

    # Calculate and print the mean and standard deviation of fold losses
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    print(f'Mean Loss: {mean_loss:.4f} (±{std_loss:.4f} std dev)')

    # Calculate and print the mean and standard deviation of F1 scores
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    print(f'Mean F1 Score: {mean_f1:.4f} (±{std_f1:.4f} std dev)')
    
    model_info = {
    'Training_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    'Optimizer' : 'adam',
    'Loss_Function' : 'binary_crossentropy',
    'Hidden_Layers' : '64-32 units',
    'Cross_Validation' : 'Stratified 5 Fold',
    'Accuracy': mean_accuracy,
    'Loss': mean_loss, 
    'F1_Score': mean_f1  
    }
    
    model_info_df = pd.DataFrame(model_info)
    
    write_to_sql(model_info_df,'customer_retention_model_information',overwrite=False)
    model.save(r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\src\models\MLP_model.keras')
    print("The model has been trained and saved successfully!")


def train(algorithm = 'MLP'): 
    data = get_training_data()
    train_xgboost(data)
    best_params = hyperparameter_tuning_MLP(data)
    train_MLP(data)
    print("training complete!")

