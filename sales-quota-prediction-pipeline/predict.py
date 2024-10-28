import pandas as pd 

import joblib
from tensorflow.keras.models import load_model
from create_training_data import prepare_inference_data,prepare_inference_data1
from connect_to_database import write_to_sql
import numpy as np
import scipy.stats as stats


def load_and_predict_ensemble(inference_data):
    # Define the feature columns
    train_columns_with_backlog = ['Lagged_EUR_std', 'Backlog_std', 'Spread_std',
                                  'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
                                  'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    train_columns_without_backlog = ['Lagged_EUR_std', 'Spread_std',
                                     'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
                                     'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    # Load the models and scalers
    model_with_backlog = load_model(f'model/model_with_backlog.h5')
    model_without_backlog = load_model(f'model/model_without_backlog.h5')
    scaler_with_backlog = joblib.load(f'model/scaler_with_backlog.pkl')
    scaler_without_backlog = joblib.load(f'model/scaler_without_backlog.pkl')
    
    # Prepare the data for inference
    X_inference_with_backlog = inference_data[train_columns_with_backlog]
    X_inference_without_backlog = inference_data[train_columns_without_backlog]
    
    # Standardize the features
    X_inference_with_backlog_scaled = scaler_with_backlog.transform(X_inference_with_backlog)
    X_inference_without_backlog_scaled = scaler_without_backlog.transform(X_inference_without_backlog)
    
    # Make predictions
    y_pred_with_backlog = model_with_backlog.predict(X_inference_with_backlog_scaled).flatten()
    y_pred_without_backlog = model_without_backlog.predict(X_inference_without_backlog_scaled).flatten()
    
    # Average the predictions from both models
    y_pred_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2
    
    return y_pred_ensemble

def load_and_predict_ensemble_0(inference_data):

    train_columns_with_backlog = ['Lagged_EUR_std', 'Backlog_std',
                                  'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
                                  'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    train_columns_without_backlog = ['Lagged_EUR_std',
                                     'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
                                     'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    model_with_backlog = load_model(f'model/model_with_backlog.h5')
    model_without_backlog = load_model(f'model/model_without_backlog.h5')
    scaler_with_backlog = joblib.load(f'model/scaler_with_backlog.pkl')
    scaler_without_backlog = joblib.load(f'model/scaler_without_backlog.pkl')
    
    
    condition = (inference_data['EUR'] == 0) & (inference_data['Amount'] == 0)
    
    
    y_pred_ensemble = pd.Series(0, index=inference_data.index)

    
    inference_data_filtered = inference_data[~condition]
    
    X_inference_with_backlog = inference_data_filtered[train_columns_with_backlog]
    X_inference_without_backlog = inference_data_filtered[train_columns_without_backlog]
    
    X_inference_with_backlog_scaled = scaler_with_backlog.transform(X_inference_with_backlog)
    X_inference_without_backlog_scaled = scaler_without_backlog.transform(X_inference_without_backlog)
    
    y_pred_with_backlog = model_with_backlog.predict(X_inference_with_backlog_scaled).flatten()
    y_pred_without_backlog = model_without_backlog.predict(X_inference_without_backlog_scaled).flatten()
    
    y_pred_filtered_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2
    
    y_pred_ensemble[~condition] = y_pred_filtered_ensemble
    
    return y_pred_ensemble


def test1(inference_data):
    # Define the feature columns
    train_columns_with_backlog = [
        'Lagged_EUR_std', 'Backlog_std',
        'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
        'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT'
    ]
    
    train_columns_without_backlog = [
        'Lagged_EUR_std',
        'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std',
        'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT'
    ]
    
    # Load the models and scalers
    model_with_backlog = load_model('model/model_with_backlog.h5')
    model_without_backlog = load_model('model/model_without_backlog.h5')
    scaler_with_backlog = joblib.load('model/scaler_with_backlog.pkl')
    scaler_without_backlog = joblib.load('model/scaler_without_backlog.pkl')
    
    # Define the conditions
    condition1 = (inference_data['EUR'] == 0) & (inference_data['Amount'] == 0)
    condition2 = (inference_data['Amount'] == 0) & (inference_data['EUR'] != 0)
    condition3 = (inference_data['EUR'] < 10000) & (~condition1) & (~condition2)
    
    # Initialize the prediction Series with NaN
    y_pred_ensemble = pd.Series(np.nan, index=inference_data.index)
    
    # Apply Condition 1: EUR == 0 and Amount == 0
    y_pred_ensemble[condition1] = 0
    
    # Apply Condition 2: Amount == 0 and EUR != 0 (Increase EUR by 2%)
    y_pred_ensemble[condition2] = inference_data.loc[condition2, 'EUR'] * 1.02
    
    # Apply Condition 3: EUR < 10,000 (Increase EUR by 1.5%)
    y_pred_ensemble[condition3] = inference_data.loc[condition3, 'EUR'] * 1.015
    
    # Define the remaining rows that do not meet any of the above conditions
    remaining_rows = ~(condition1 | condition2 | condition3)
    
    if remaining_rows.any():
        # Filter the inference data for remaining rows
        inference_data_filtered = inference_data[remaining_rows]
        
        # Select the appropriate feature columns
        X_inference_with_backlog = inference_data_filtered[train_columns_with_backlog]
        X_inference_without_backlog = inference_data_filtered[train_columns_without_backlog]
        
        # Standardize the features using the respective scalers
        X_inference_with_backlog_scaled = scaler_with_backlog.transform(X_inference_with_backlog)
        X_inference_without_backlog_scaled = scaler_without_backlog.transform(X_inference_without_backlog)
        
        # Make predictions using both models
        y_pred_with_backlog = model_with_backlog.predict(X_inference_with_backlog_scaled).flatten()
        y_pred_without_backlog = model_without_backlog.predict(X_inference_without_backlog_scaled).flatten()
        
        # Average the predictions from both models
        y_pred_filtered_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2
        
        # Assign the ensemble predictions to the remaining rows
        y_pred_ensemble[remaining_rows] = y_pred_filtered_ensemble
    
    
    return y_pred_ensemble


def test2(inference_data):
    # Define the feature columns
    train_columns_with_backlog = ['Lagged_EUR_std', 'Backlog_std',
                                  'Lagged_Quota_std', 'Lagged_Monetary_std' ,'Lagged_Frequency_std',
                                  'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    train_columns_without_backlog = ['Lagged_EUR_std',
                                     'Lagged_Quota_std','Lagged_Monetary_std', 'Lagged_Frequency_std',
                                     'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT']
    
    # Load the models and scalers
    model_with_backlog = load_model(f'model/model_with_backlog.h5')
    model_without_backlog = load_model(f'model/model_without_backlog.h5')
    scaler_with_backlog = joblib.load(f'model/scaler_with_backlog.pkl')
    scaler_without_backlog = joblib.load(f'model/scaler_without_backlog.pkl')
    
    # Condition to check EUR == 0 and Amount == 0
    condition_zero_eur_and_amount = (inference_data['EUR'] == 0) & (inference_data['Amount'] == 0)
    
    # Initialize predictions with zeros for rows that meet the condition
    y_pred_ensemble = pd.Series(0, index=inference_data.index)
    
    # Prepare the data for inference for rows that do not meet the zero EUR and Amount condition
    inference_data_filtered = inference_data[~condition_zero_eur_and_amount]
    
    X_inference_with_backlog = inference_data_filtered[train_columns_with_backlog]
    X_inference_without_backlog = inference_data_filtered[train_columns_without_backlog]
    
    # Standardize the features
    X_inference_with_backlog_scaled = scaler_with_backlog.transform(X_inference_with_backlog)
    X_inference_without_backlog_scaled = scaler_without_backlog.transform(X_inference_without_backlog)
    
    # Make predictions
    y_pred_with_backlog = model_with_backlog.predict(X_inference_with_backlog_scaled).flatten()
    y_pred_without_backlog = model_without_backlog.predict(X_inference_without_backlog_scaled).flatten()
    
    # Average the predictions from both models
    y_pred_filtered_ensemble = (y_pred_with_backlog + y_pred_without_backlog) / 2
    
    # Assign predictions back to the appropriate rows
    y_pred_ensemble[~condition_zero_eur_and_amount] = y_pred_filtered_ensemble
    
    # New condition: If growth exceeds 50%, cap the growth to 1-3%
    growth_condition = ((y_pred_ensemble - inference_data['Amount']) / inference_data['Amount']) > 0.12
    growth_condition2 = ((3*inference_data['EUR']) < inference_data['Amount'])

    capped_growth = inference_data['Amount'][growth_condition] * (1 + np.random.uniform(0.03, 0.08))
    capped_growth2 = inference_data['EUR'][growth_condition2] * (1 + np.random.uniform(0.03 ,0.08))

    y_pred_ensemble[growth_condition] = capped_growth
    y_pred_ensemble[growth_condition2] = capped_growth2
    
    return y_pred_ensemble

def post_process(preds, confidence_level=0.25, optimization=1):
    new_preds = []
    for p in preds:
        # Calculate mean squared error (MSE) from y_true and y_pred
        y_true = p['y_true']
        y_pred = p['y_pred']
        mse = np.mean((y_true - y_pred)**2)
        
        # Calculate confidence interval based on MSE and the provided formula
        std = p['Spread_std']
        n = len(preds)  # Assuming the number of observations

        # Calculate sum of squares of x (assuming x is available in your data)
        x = p['Lagged_Quota_std']  # Replace 'x' with the actual predictor variable from your data
        x_mean = np.mean(x)
        SS_x = np.sum((x - x_mean)**2)

        # Degrees of freedom
        df = n - 1

        # Critical value from t-distribution based on confidence level and df
        t = stats.t.ppf((1 + confidence_level) / 2, df)

        q75_backlog = p['Backlog_std'].quantile(0.75)
        q25_backlog = p['Backlog_std'].quantile(0.25)
        
        # Calculate confidence interval using the formula
        CI_range = t * np.sqrt(mse * (1 + 1/n + ((x - x_mean)**2 / SS_x)))
        p['y_upper'] = p['y_pred'] + CI_range
        p['y_lower'] = p['y_pred'] - CI_range

        p.loc[p['Backlog_std'] >= q75_backlog, 'y_upper'] += 0.05 * p['y_pred']

        p.loc[p['Backlog_std'] <= q25_backlog, 'y_lower'] += 0.05 * p['y_pred']
       # Dynamic adjustment of prediction based on optimization factor
        if optimization != 0:
            opt_factor = optimization * (p['y_upper'] - p['y_pred'])
            p['y_adjusted'] = p['y_pred'] + opt_factor
        else:
            p['y_adjusted'] = p['y_pred']

        new_preds.append(p)

    return new_preds
def get_growth_adjustment_weight(growth_potential):
    if growth_potential == 'High':
        return 1.4
    elif growth_potential == 'Medium':
        return 1.2
    else:
        return 1


def predict():
    data = prepare_inference_data1()
    y_pred = test2(data)
    data['Predicted_Quota'] = y_pred
    data['Inactive_OR_No_Quota'] = ((data['Amount'] == 0) | (data['EUR'] == 0))
    
    data['growth_adjustment_weight'] = data['Growth_Potential'].apply(get_growth_adjustment_weight)

    data = data.rename(columns={'EUR' : 'Current_Year_Revenue', 
                 'Amount' : 'Current_Year_Quota'})
    sql_df = data[['sales_id','SalesName','Current_Year_Revenue','CAGR','Current_Year_Quota','Predicted_Quota','Inactive_OR_No_Quota','Master_Sector','Fact_Entity','Growth_Potential','Month','year_id','growth_adjustment_weight']]
    sql_df.to_csv('Predicted.csv',index=False,header=True)
    print(f"Total predicted Quota : {data['Predicted_Quota'].sum()}")
    print(f"Total Projected Revenue : {data['Current_Year_Revenue'].sum()}")
    print(f"Current Year Quota : {data['Current_Year_Quota'].sum()}")
    write_to_sql(sql_df,'ai_sales_quota_predictions')
predict()