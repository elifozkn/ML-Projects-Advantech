import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model
import joblib
from create_training_data import prepare
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
def train_and_evaluate_nn(df_encoded):
    # Define lists to store metrics for each iteration
    mse_values = []
    mae_values = []
    mape_values = []
    r2_values = []  # List to store R^2 values
    year_dfs = []

    # Define the range of years from 2021 to 2025
    years = range(2021, 2025)

    for i in range(len(years)-1):
        # Filter data for training and testing based on the years up to the current iteration
        train_data = df_encoded[df_encoded['year_id'].isin(years[:i+1])]
        train_data = train_data[train_data['Lagged_Quota_std'] != 0]
        train_data = train_data[train_data['Lagged_EUR_std'] != 0]
        test_data = df_encoded[df_encoded['year_id'] == years[i+1]]
        train_columns = ['Lagged_EUR_std','Lagged_Partial_Win_Rate','Spread_std','Lagged_Quota_std','Lagged_Monetary_std','Lagged_Frequency_std','Month','CIOT', 'EIoT', 'IIoT', 'SIOT']    
        # Separate features and target for training and testing
        X_train = train_data[train_columns]
        y_train = train_data['Amount']
        X_test = test_data[train_columns]
        y_test = test_data['Amount']
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1)  # Output layer with single neuron for regression
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Predict on the testing data
        y_pred = model.predict(X_test_scaled).flatten()
        
        # Compute evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)
        
        mae = mean_absolute_error(y_test, y_pred)
        mae_values.append(mae)
        
        mape = np.median(np.abs((y_test - y_pred) / (y_test))) * 100
        mape_values.append(mape)
        
        r2 = r2_score(y_test, y_pred)
        r2_values.append(r2)

        print(f"Mean Squared Error for predicting {years[i+1]} based on data from years {years[:i+1]}: {mse}")
        print(f"Mean Absolute Error for predicting {years[i+1]} based on data from years {years[:i+1]}: {mae}")
        print(f"Mean Absolute Percentage Error for predicting {years[i+1]} based on data from years {years[:i+1]}: {mape}")
        print(f"R^2 score for predicting {years[i+1]} based on data from years {years[:i+1]}: {r2}")
        
        # Store predictions and true values in a DataFrame
        df_with_predictions = X_test.copy()
        df_with_predictions['y_true'] = y_test
        df_with_predictions['y_pred'] = y_pred

        percentage_error = np.abs((y_test - y_pred) / y_test) * 100
        df_with_predictions['Percentage_Error'] = percentage_error
        df_with_predictions['sales_id'] = test_data['sales_id']
        df_with_predictions['year_id'] = test_data['year_id']
        df_with_predictions['Historical_Mean'] = test_data['Historical_Mean']
        df_with_predictions['Lagged_Quota'] = test_data['Lagged_Quota']
        df_with_predictions['Lagged_EUR'] = test_data['Lagged_EUR']
        df_with_predictions['Spread'] = test_data['Spread']

        year_dfs.append(df_with_predictions)
        
        # Compute SHAP values using GradientExplainer
        background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Plot SHAP values
        shap.summary_plot(shap_values, X_test, feature_names=train_columns,plot_type='bar')
        plt.show()

    return year_dfs


def train_and_evaluate_nn_ensemble(df_encoded, test_size=0.2, random_state=42):
    # Filter out rows with zero in the lagged columns
    df_filtered = df_encoded#[(df_encoded['Lagged_Quota_std'] != 0) & (df_encoded['Lagged_EUR_std'] != 0)]

    # Feature columns
    train_columns_with_backlog = ['Lagged_EUR_std', 'Backlog_std', 'Lagged_Quota_std', 
                                  'Lagged_Monetary_std','Lagged_Opportunity_Amount', 'Lagged_Frequency_std', 'Month', 'CIOT', 
                                  'EIoT', 'IIoT', 'SIOT']
    
    train_columns_without_backlog = ['Lagged_EUR_std', 'Lagged_Quota_std', 'Lagged_Opportunity_Amount','Lagged_Monetary_std', 
                                     'Lagged_Frequency_std', 'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']

    # Separate features and target
    X_with_backlog = df_filtered[train_columns_with_backlog]
    X_without_backlog = df_filtered[train_columns_without_backlog]
    y = df_filtered['Amount']

    # Train-test split
    X_train_with_backlog, X_test_with_backlog, y_train, y_test = train_test_split(X_with_backlog, y, 
                                                                                  test_size=test_size, 
                                                                                  random_state=random_state)
    X_train_without_backlog, X_test_without_backlog, _, _ = train_test_split(X_without_backlog, y, 
                                                                             test_size=test_size, 
                                                                             random_state=random_state)

    # Standardize features
    scaler_with_backlog = StandardScaler()
    X_train_with_backlog_scaled = scaler_with_backlog.fit_transform(X_train_with_backlog)
    X_test_with_backlog_scaled = scaler_with_backlog.transform(X_test_with_backlog)

    scaler_without_backlog = StandardScaler()
    X_train_without_backlog_scaled = scaler_without_backlog.fit_transform(X_train_without_backlog)
    X_test_without_backlog_scaled = scaler_without_backlog.transform(X_test_without_backlog)

    def build_and_train_model(X_train_scaled, y_train):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1)  # Output layer for regression
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        return model

    # Train models
    model_with_backlog = build_and_train_model(X_train_with_backlog_scaled, y_train)
    model_without_backlog = build_and_train_model(X_train_without_backlog_scaled, y_train)

    # Save models and scalers
    save_model(model_with_backlog, 'model/model_with_backlog.h5')
    save_model(model_without_backlog, 'model/model_without_backlog.h5')
    joblib.dump(scaler_with_backlog, 'model/scaler_with_backlog.pkl')
    joblib.dump(scaler_without_backlog, 'model/scaler_without_backlog.pkl')

    # Predict on evaluation data
    y_pred_with_backlog = model_with_backlog.predict(X_test_with_backlog_scaled).flatten()
    y_pred_without_backlog = model_without_backlog.predict(X_test_without_backlog_scaled).flatten()

    # Ensemble prediction
    y_pred_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred_ensemble)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    mape = np.median(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
    r2 = r2_score(y_test, y_pred_ensemble)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print(f"R^2 score: {r2}")

    # Compute percentage errors
    percentage_errors = np.abs((y_test - y_pred_ensemble) / y_test) * 100

    # Count the number of instances in each MAPE range
    under_10 = np.sum(percentage_errors < 10)
    between_10_and_25 = np.sum((percentage_errors >= 10) & (percentage_errors < 25))
    between_25_and_50 = np.sum((percentage_errors >= 25) & (percentage_errors < 50))
    above_50 = np.sum(percentage_errors >= 50)

    print(f"Number of instances with MAPE < 10%: {under_10}")
    print(f"Number of instances with MAPE between 10% and 25%: {between_10_and_25}")
    print(f"Number of instances with MAPE between 25% and 50%: {between_25_and_50}")
    print(f"Number of instances with MAPE >= 50%: {above_50}")

    # Optionally return metrics and counts if needed
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'counts': {
            'under_10': under_10,
            'between_10_and_25': between_10_and_25,
            'between_25_and_50': between_25_and_50,
            'above_50': above_50
        }
    }



def build_model(optimizer='adam', neurons=128, dropout_rate=0.5, l2_reg=0.01):
    
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train_with_backlog_scaled.shape[1],), kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(neurons // 4, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_evaluate_nn_ensemble(df_encoded, test_size=0.2, random_state=42):
    # Filter and define features and target
    df_filtered = df_encoded

    train_columns_with_backlog = ['Lagged_EUR_std', 'Backlog_std', 'Lagged_Quota_std', 
                                  'Lagged_Monetary_std','Lagged_Opportunity_Amount', 'Lagged_Frequency_std', 'Month', 'CIOT', 
                                  'EIoT', 'IIoT', 'SIOT']
    train_columns_without_backlog = ['Lagged_EUR_std', 'Lagged_Quota_std', 'Lagged_Opportunity_Amount','Lagged_Monetary_std', 
                                     'Lagged_Frequency_std', 'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']

    X_with_backlog = df_filtered[train_columns_with_backlog]
    X_without_backlog = df_filtered[train_columns_without_backlog]
    y = df_filtered['Amount']

    X_train_with_backlog, X_test_with_backlog, y_train, y_test = train_test_split(X_with_backlog, y, 
                                                                                  test_size=test_size, 
                                                                                  random_state=random_state)
    X_train_without_backlog, X_test_without_backlog, _, _ = train_test_split(X_without_backlog, y, 
                                                                             test_size=test_size, 
                                                                             random_state=random_state)

    # Standardize features
    scaler_with_backlog = StandardScaler()
    X_train_with_backlog_scaled = scaler_with_backlog.fit_transform(X_train_with_backlog)
    X_test_with_backlog_scaled = scaler_with_backlog.transform(X_test_with_backlog)

    scaler_without_backlog = StandardScaler()
    X_train_without_backlog_scaled = scaler_without_backlog.fit_transform(X_train_without_backlog)
    X_test_without_backlog_scaled = scaler_without_backlog.transform(X_test_without_backlog)

    # Create the Keras Regressor wrapper for hyperparameter tuning
    model_with_backlog = KerasRegressor(build_fn=build_model, verbose=0)
    model_without_backlog = KerasRegressor(build_fn=build_model, verbose=0)

    # Define the hyperparameters to search over
    param_grid = {
        'neurons': [32,64, 128, 256,512],
        'dropout_rate': [0.3, 0.5, 0.7],
        'optimizer': ['adam', 'rmsprop', Adam(learning_rate=0.001)],
        'l2_reg': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100]
    }

    # Randomized search over hyperparameters
    random_search_with_backlog = RandomizedSearchCV(estimator=model_with_backlog, param_distributions=param_grid,
                                                    n_iter=5, cv=3, verbose=1, n_jobs=-1)
    random_search_without_backlog = RandomizedSearchCV(estimator=model_without_backlog, param_distributions=param_grid,
                                                       n_iter=5, cv=3, verbose=1, n_jobs=-1)

    # Fit the models
    random_search_with_backlog.fit(X_train_with_backlog_scaled, y_train)
    random_search_without_backlog.fit(X_train_without_backlog_scaled, y_train)

    # Best models
    best_model_with_backlog = random_search_with_backlog.best_estimator_.model
    best_model_without_backlog = random_search_without_backlog.best_estimator_.model

    # Save models and scalers
    save_model(best_model_with_backlog, 'model/model_with_backlog.h5')
    save_model(best_model_without_backlog, 'model/model_without_backlog.h5')
    joblib.dump(scaler_with_backlog, 'model/scaler_with_backlog.pkl')
    joblib.dump(scaler_without_backlog, 'model/scaler_without_backlog.pkl')

    # Predict on evaluation data
    y_pred_with_backlog = best_model_with_backlog.predict(X_test_with_backlog_scaled).flatten()
    y_pred_without_backlog = best_model_without_backlog.predict(X_test_without_backlog_scaled).flatten()

    # Ensemble prediction
    y_pred_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred_ensemble)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    mape = np.median(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
    r2 = r2_score(y_test, y_pred_ensemble)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print(f"R^2 score: {r2}")

    # Compute percentage errors
    percentage_errors = np.abs((y_test - y_pred_ensemble) / y_test) * 100

    # Count the number of instances in each MAPE range
    under_10 = np.sum(percentage_errors < 10)
    between_10_and_25 = np.sum((percentage_errors >= 10) & (percentage_errors < 25))
    between_25_and_50 = np.sum((percentage_errors >= 25) & (percentage_errors < 50))
    above_50 = np.sum(percentage_errors >= 50)

    print(f"Number of instances with MAPE < 10%: {under_10}")
    print(f"Number of instances with MAPE between 10% and 25%: {between_10_and_25}")
    print(f"Number of instances with MAPE between 25% and 50%: {between_25_and_50}")
    print(f"Number of instances with MAPE >= 50%: {above_50}")

    # Optionally return metrics and counts if needed
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'counts': {
            'under_10': under_10,
            'between_10_and_25': between_10_and_25,
            'between_25_and_50': between_25_and_50,
            'above_50': above_50
        }
    }


def count_forecasting_categories(df):
    # Categorize instances based on MAPE values
    highly_accurate = df[df['Percentage_Error'] < 10]
    good_forecasting = df[(df['Percentage_Error'] >= 10) & (df['Percentage_Error'] < 25)]
    reasonable_forecasting = df[(df['Percentage_Error'] >= 25) & (df['Percentage_Error'] < 50)]
    weak_inaccurate = df[df['Percentage_Error'] >= 50]
    
    # Count number of instances in each category
    counts = {
        'Highly accurate prediction': len(highly_accurate),
        'Good forecasting': len(good_forecasting),
        'Reasonable forecasting': len(reasonable_forecasting),
        'Weak and inaccurate forecasting': len(weak_inaccurate)
    }

    highly_accurate_sales_id = highly_accurate['sales_id'].unique()
    good_forecasting_sales_id = good_forecasting['sales_id'].unique()
    reasonable_forecasting_sales_id = reasonable_forecasting['sales_id'].unique()
    weak_inaccurate_sales_id = weak_inaccurate['sales_id'].unique()

    print(f"MAPE without the weak predictions : {df[df['Percentage_Error'] < 70]['Percentage_Error'].mean()}" )
    print(f"# of sales id highly accurate {len(highly_accurate_sales_id)}")
    print(f"# of sales id good forecasting {len(good_forecasting_sales_id)}")
    print(f"# of sales id reasonable forecasting {len(reasonable_forecasting_sales_id)}")
    print(f"# of sales id weak and inaccurate {len(weak_inaccurate_sales_id)}")
    return counts,highly_accurate,good_forecasting,reasonable_forecasting,weak_inaccurate

df = prepare()
train_and_evaluate_nn_ensemble(df)
