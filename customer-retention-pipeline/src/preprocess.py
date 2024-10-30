import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from connect_to_database import write_to_sql
from datetime import datetime as dt, timedelta

CATEGORICAL_COLUMNS = ['egroup', 'cust_country','Label','Master Sector','Customer','Customer_Id','order_date_list','erp_id', 'close_date', 'backlog_date', 'Customer_Id_y']

NUMERICAL_VARS = ['Recency', 'Frequency', 'Monetary',
       'Recency_Score', 'Frequency_Score', 'Monetary_Score', 'RFM_Score',
       'qty','Opportunity_Amt','Backlog']

def smart_undersample(df):
    # Extract features (X) and labels (y) from the DataFrame
    X = df.copy()
    X = df.drop(columns=['target_encoded'],axis =1) 
    y = df['target_encoded']

    # Calculate class label ratios
    class_counts = np.bincount(y)
    majority_class_count = np.max(class_counts)
    minority_class_count = np.min(class_counts)
    ratio = majority_class_count / len(y)
    # Check if the ratio is larger than 4/6 (2/3)
    if ratio > 0.6:
        # Find Tomek links
        nn = NearestNeighbors(n_neighbors=2)
        fit_X =X.drop(columns = ['Label','Customer_Id','cust_country'], axis =1)
        nn.fit(fit_X)
        neighbors = nn.kneighbors(return_distance=False)
        tomek_indices = []

        for i in range(len(fit_X)):
            if y.iloc[i] == np.argmax(class_counts):  # Majority class
                if y.iloc[neighbors[i][1]] != np.argmax(class_counts):  # Minority class neighbor
                    tomek_indices.append(i)

        # Remove Tomek links from the majority class
        majority_indices = np.where(y == np.argmax(class_counts))[0]
        undersampled_majority_indices = np.setdiff1d(majority_indices, tomek_indices)

        # Combine the undersampled majority class with the minority class
        undersampled_df = df.iloc[np.concatenate([undersampled_majority_indices, np.where(y != np.argmax(class_counts))[0]])]
        undersampled_df = undersampled_df.reset_index(drop = True)

        majority_class = undersampled_df[undersampled_df['target_encoded'] == 1]
        minority_class = undersampled_df[undersampled_df['target_encoded'] == 0]
        # Randomly undersample the majority class to match the minority class size
        #majority_class_undersampled = resample(majority_class, n_samples=minority_class_count, random_state=42)
        
        #balanced_data = pd.concat([majority_class_undersampled, minority_class])
        #balanced_data = balanced_data.sample(frac=1, random_state=42)

        return undersampled_df
    else:
        # Return the original DataFrame if no undersampling is needed
        return df

def remove_outliers_z(data, columns, threshold=2):

    """
    Remove outliers from the dataset using the Z-score method.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to check for outliers.
        threshold (float): Threshold for identifying outliers in terms of Z-score. Default is 3.
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    filtered_data = data.copy()
    
    for column in columns:
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        filtered_data = filtered_data[z_scores <= threshold]
        filtered_data = filtered_data.reset_index()
    
    return filtered_data

def standardize(df,columns) : 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    X_scaled_dataframe = pd.DataFrame(X_scaled, columns=columns)

    combined_df = pd.concat([X_scaled_dataframe, df[CATEGORICAL_COLUMNS]], axis=1)
    return combined_df

def one_hot_encode(df, columns_to_encode):
   
    encoded_df = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)
    return encoded_df


def encode_categorical_target(df, target_column):
    """
    Encode a categorical target variable into numerical labels using LabelEncoder.
    
    Parameters:
    - df: DataFrame
        The input DataFrame containing the target column.
    - target_column: str
        The name of the categorical target column.
    
    Returns:
    - df_encoded: DataFrame
        A new DataFrame with the target column encoded as 'target_encoded'.
    - label_encoder: LabelEncoder
        The LabelEncoder object used for encoding.
    """
    # Create a copy of the input DataFrame to avoid overwriting
    df_encoded = df.copy()
    
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()
    
    # Encode the target column and add it as 'target_encoded'
    df_encoded['target_encoded'] = label_encoder.fit_transform(df_encoded[target_column])
    
    return df_encoded, label_encoder


def remove_unneccesary_cols(df,sector = "Customer",pred = False):
    if pred : 
        df = df.drop(['Recency','Frequency','Monetary','egroup',
                  'Recency_Score','order_date_list',sector
                  ,'erp_id', 'close_date', 'backlog_date', 'Customer_Id_y'],axis =1 )

    else : 
        df = df.drop(['Recency','Frequency','Monetary','egroup',
                  'Recency_Score',sector
                  ,'erp_id', 'close_date', 'backlog_date', 'Customer_Id_y'],axis =1 )
    return df 

def drop_old_customers(df): 
    two_years_ago = dt(2021, 1, 1)
    # Check the condition for each row
    for index, row in df.iterrows():
        order_date =(max(row['order_date_list']))
        if order_date < two_years_ago:
            df = df.drop(index)

    # Reset the index of the DataFrame to make it continuous
    df = df.reset_index(drop=True)

    return df

def preprocess(data,pred = False): 
    print(data.isna().any())
    if pred : 
        data = standardize(data,NUMERICAL_VARS)
        data = one_hot_encode(data,['Customer'])
        #data = drop_old_customers(data)
        data = remove_unneccesary_cols(data,sector = "Master Sector",pred=pred)
        data = data.rename({'\'Customer_KA, CP, AOL\'' : 'Customer_All'})
        write_to_sql(data,'customer_retention_to_be_predicted')
        print('inference data has been created successfully. ')
        return data
    else: 
        data = remove_outliers_z(data,['Monetary'])
        data = standardize(data,NUMERICAL_VARS)
        data = one_hot_encode(data,['Customer'])
        data,le = encode_categorical_target(data,'Label')
        data = remove_unneccesary_cols(data,sector = 'Master Sector',pred=True)
        data = smart_undersample(data)
        data = data.rename({'Customer_KA, CP, AOL' : 'Customer_All'})
        write_to_sql(data,'customer_retention_training_data')
        print('training data has been created successfully. ')
        return data