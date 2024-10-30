
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
from post_process import post_process
from datetime import date
import numpy as np
import pandas as pd

from connect_to_database import write_to_sql
from connect_to_database import get_to_be_predicted_data
def predict_next_year_MLP(): 
    #todo : add data reading from the database
    data = get_to_be_predicted_data()
    # Load the saved model
    loaded_model = keras.models.load_model(r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\src\models\MLP_2022.keras')
    output_data = data.copy() 

    X_new = data.drop(['Label','Customer_Id','cust_country'],axis = 1)

    scaler = StandardScaler()
    #X_new = scaler.fit_transform(X_new)

    # Make predictions on the new dataset
    predictions = loaded_model.predict(X_new)
    # Round predictions to obtain binary labels (0 or 1)
    #binary_predictions = (predictions > 0.5).astype(int)

    # Use 'predict' method to get raw probabilities
    raw_probabilities = loaded_model.predict(X_new)
    #print(raw_probabilities)
    #todo : search for the binary cross entropy, if it has an affect
    probability_class_0 = 1 - raw_probabilities
    output_data['Predictions'] = np.where(raw_probabilities >= 0.5, 0, 1)

    # Add columns for raw probabilities
    output_data['Probability_Class_0'] = probability_class_0  # Probability for class 0
    output_data['Probability_Class_1'] = raw_probabilities  # Probability for class 1
    output_file_path = r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\test\prediction_results\prediction_results_latest_MLP.csv'
    output_data.to_csv(output_file_path, index=False)  # Set index=False to avoid writing row numbers
    
    print(f'Combined data with predictions and probabilities saved to {output_file_path}')


def predict_with_xgboost() : 
    df = get_to_be_predicted_data()
    model_filename = r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\src\models\xgboost.model'
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_filename)  # Replace with the path to your trained model file

  
    new_data = df
    X_new = new_data.drop(['Label','Customer_Id','cust_country'],axis = 1)
    X_new['Customer_AOL'] = X_new['Customer_AOL'].astype(int)
    X_new['Customer_CP'] = X_new['Customer_CP'].astype(int)
    X_new['Customer_KA'] = X_new['Customer_KA'].astype(int)

    X_new['Customer_AOL'] = X_new['Customer_AOL'].astype(bool)
    X_new['Customer_CP'] = X_new['Customer_CP'].astype(bool)
    X_new['Customer_KA'] = X_new['Customer_KA'].astype(bool)
    
    y_pred = loaded_model.predict(X_new)

    output_data = new_data.copy()
    class_probabilities = loaded_model.predict_proba(X_new)
    
    output_data['Predictions'] = y_pred
    output_data['Probability'] = np.where(y_pred == 1, class_probabilities[:, 1], class_probabilities[:, 0])
    output_data = post_process(output_data)
    

    write_to_sql(output_data,'customer_retention_prediction_results')

    return output_data


def predict():
    predict_with_xgboost()
    #predict_next_year_MLP()