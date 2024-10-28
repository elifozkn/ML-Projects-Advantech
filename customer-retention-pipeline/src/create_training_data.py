#import the libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler
from sklearn.cluster import DBSCAN
from datetime import timedelta
from connect_to_database import get_opportunity_data,get_data,get_historical_backlog


def convert_to_datetime(date_columns,df):

    for column in date_columns: 
        df[column] = pd.to_datetime(df[column], format="%Y-%m-%d %H:%M:%S.%f")

def convert_to_float(df):
    columns_to_convert_to_float = ['EUR', 'us_amt','qty']

    # Loop through each column and convert its values to float
    for column in columns_to_convert_to_float:
        df[column] = pd.to_numeric(df[column], errors='coerce')

'''
categorical_variables = ['Tran_Type', 'Efftive_Date', 'Order_Date', 'Order_No', 'PO', 'Region',
       'RBU', 'Sector', 'MasterSector', 'Site', 'Customer_Id', 'Customer_Name',
       'Cust_Country', 'Sales_ID', 'Sales_Name', 'Part', 'PG', 'PD', 'PDL']

numeric_variables = ['Qty', 'Us_Amt', 'EUR']

'''
def absolute_value(numeric_columns,df):
    for var in numeric_columns:
        df[var] = df[var].abs()


def calculate_rfm(data,r_coef = 1,f_coef =1,m_coef= 1):
  '''

  function that calculates the Recency Frequency Monetary Score given a dataframe.
  overall RFM score is calculated so that all components (R,F,M) have equal weights.

  return : a dataframe containing Customer RFM information.

  '''
  # Calculate the most recent purchase date for each customer
  recency_df = data.groupby('Customer_Id')['order_date'].max().reset_index()
  # Calculate the number of days since the most recent purchase (assuming today is the last date in the dataset)
  recency_df['Recency'] = (pd.to_datetime('today') - pd.to_datetime(recency_df["order_date"])).dt.days
  #recency_df.drop('Order_Date', axis=1, inplace=True)

  # Count the number of purchases for each customer
  frequency_df = data.groupby('Customer_Id')['order_date'].count().reset_index()
  frequency_df.rename(columns={'order_date': 'Frequency'}, inplace=True)

  # Calculate the total monetary value of purchases for each customer
  monetary_df = data.groupby('Customer_Id')['EUR'].sum().reset_index()
  monetary_df.rename(columns={'EUR': 'Monetary'}, inplace=True)

  # Merge the recency, frequency, and monetary DataFrames into a single RFM DataFrame
  rfm_df = pd.merge(recency_df, frequency_df, on='Customer_Id')
  rfm_df = pd.merge(rfm_df, monetary_df, on='Customer_Id')

  rfm_df['Recency_Score'] = pd.qcut(rfm_df['Recency'], q=4, labels=False)
  rfm_df['Frequency_Score'] = pd.qcut(rfm_df['Frequency'], q=5, labels=False,duplicates = 'drop')
  rfm_df['Monetary_Score'] = pd.qcut(rfm_df['Monetary'], q=4, labels=False)

  # Calculate the RFM score by combining the individual scores
  rfm_df['RFM_Score'] = -(rfm_df['Recency_Score'])*r_coef + rfm_df['Frequency_Score']*f_coef + rfm_df['Monetary_Score']*m_coef
  #print(rfm_df.head())

  return rfm_df

def merge_rfm(df,rfm_df):
    merged = pd.merge(df, rfm_df, on = "Customer_Id")
    column_index_to_drop = 3  # Index of the column you want to drop
    df = df.drop(df.columns[column_index_to_drop], axis=1)
    # these columns do not have much variation, or not important for our class
    cols_to_drop = ['tran_type', 'fact_1234', 'bomseq', 'gameFLAG', 'fact_zone',
       'fact_entity', 'Sales_id',
       'edivision', 'acl_pdl', 'item_no', 'efftive_date', 'sale_type',
       'breakdown', 'itp_find', 'cancel_flag', 'us_amt', 'TWD',
       'RMB', 'JPY', 'SGD', 'AUD', 'MYR', 'KRW', 'BRL', 'THB', 'BUsector',
       'record_id', 'tr_entity', 'Ship_Site', 'fact_country',
       'us_cost', 'us_ex_rate', 'order_no', 'sales_site_id',
       'idh_inv_nbr', 'sam_sw', 'EndCust', 'splitFLAG', 'Site_ID', 'WebAccess',
       'PGI_Date', 'FirstDate', 'YesNo', 'alongWO', 'FromCost', 'm_type',
       'VM2']

    merged = merged.drop(cols_to_drop, axis = 1)
    return merged


def group_by_customer(merged): 
    grouped_by_cust = merged.groupby("Customer_Id")
    final_df = grouped_by_cust.agg({ "cust_country" : "first",
                                "Customer_Id" : "first",
                                 "Recency" : 'mean',
                                 "Frequency" : "mean",
                                 "Monetary" : "mean",
                                 "Recency_Score" : "mean",
                                 "Frequency_Score" : 'mean',
                                 "Monetary_Score" : "mean",
                                 "RFM_Score" : "mean",
                                 "qty" : "sum",
                                 "egroup":"first",
                                 "order_date_x" : "max",
                                 'Sector' : "first",
                                 'Purchase_Variability':'mean',
                                 'Purchase_Diversity' : 'mean',
                                 'Label' : 'first',
                                 'order_date_list' : 'first',
                                 'cust_country' : 'first'
                        
    })

    final_df = final_df.reset_index(drop= True)
    final_df.rename(columns={'order_date_x': 'order_date'}, inplace=True)
    return final_df


def categorize_churn_status(df,current_year = datetime.datetime.now().year):
    # Convert 'order_date' column to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Group the data by 'Customer_Id' and aggregate order dates into a list
    grouped = df.groupby('Customer_Id')['order_date'].agg(list).reset_index()

    # Create a function to categorize customers
    def categorize_customer(row):

        order_dates = row['order_date']
        # get the last purchase date
        last_purchase = max(order_dates)
        #now exclude the last purchase date to get the previous last date 
        list_without_max = [x for x in order_dates if x != last_purchase]

        #if the list is empty we look at the date, if it is older than 1 year from the current date, 
        # we assign churn, if it is recent enough, we assign as a new customer. 
        if (len(list_without_max) > 0): 
            previous_last_purchase = max(list_without_max)
        elif (dt(current_year,1,1)-last_purchase <= timedelta(days = 360) ) : 
            return "retain"
        else: 
            return "churn"
            
        #lastly, if the list is unempty, that means we have a previous last purchase date, 
        #we assign churn if the prev date is older than last date, otherwise we assign retain. 
        if (((last_purchase - previous_last_purchase) > timedelta(days = 365))):
            return 'churn'
        if((dt(current_year,1,1)-previous_last_purchase) > timedelta(days = 480)):
            return'churn'
        else:
            return 'retain'
    
    grouped['Label'] = grouped.apply(categorize_customer, axis=1)
    grouped.rename(columns = {"order_date" : "order_date_list"}, inplace = True)

    return grouped



def calculate_purchase_variability(data_frame):
    data_frame['order_date'] = pd.to_datetime(data_frame['order_date'])

    # Sort data by 'Customer_Id' and 'order_date'
    data_frame = data_frame.sort_values(by=['Customer_Id', 'order_date'])

    # Calculate time intervals between purchases for each customer
    data_frame['Time_Between_Purchases'] = data_frame.groupby('Customer_Id')['order_date'].diff().dt.days

    # Calculate purchase variability (standard deviation of time intervals)
    purchase_variability = data_frame.groupby('Customer_Id')['Time_Between_Purchases'].std()

    # Merge purchase variability back to the DataFrame
    data_frame['Purchase_Variability'] = data_frame['Customer_Id'].map(purchase_variability)

    # Fill missing purchase variability values with the mean
    mean_purchase_variability = data_frame['Purchase_Variability'].mean()
    data_frame['Purchase_Variability'].fillna(mean_purchase_variability, inplace = True)

    return data_frame

def fix_year(date_str):
    if date_str.startswith('3024'):
        return date_str.replace('3024', '2024', 1)
    return date_str

def opportunity(): 
    opp = get_opportunity_data()
    opp = opp[['erp_id', 'amount', 'close_date','probability']]
    # Apply the fix
    opp['close_date'] = opp['close_date'].apply(fix_year)

    # Convert to datetime after fixing
    opp['close_date'] = pd.to_datetime(opp['close_date'], format="%Y-%m-%d", errors='coerce')
 
    opp = opp.loc[opp['probability'] != 0] 
    opp = opp.groupby('erp_id').agg({'amount' : 'sum', 
                                     'close_date' : 'max', 
                                     }).reset_index()
    print(opp.columns)
    return opp

def backlog(current_backlog):
    backlog = get_historical_backlog()
    backlog['efftive_date'] = pd.to_datetime(backlog['efftive_date'], format="%Y-%m-%d", errors='coerce')
    current_backlog['efftive_date'] = pd.to_datetime(current_backlog['efftive_date'], format="%Y-%m-%d", errors='coerce')
    
    backlog = backlog.groupby('Customer_ID').agg({'EUR':'sum',
                                                  'efftive_date' : 'max'}).reset_index()
    
    current_backlog = current_backlog.groupby('Customer_Id').agg({'EUR':'sum',
                                                  'efftive_date' : 'max'}).reset_index()
    backlog.rename(columns={'EUR':'Backlog',
                            'efftive_date' : 'backlog_date'},inplace=True)
    current_backlog.rename(columns={'EUR':'Backlog',
                            'efftive_date' : 'backlog_date'},inplace=True) 
    backlog = pd.concat([backlog,current_backlog], ignore_index=True)   
    print(backlog.columns)
    return backlog

def calculate_purchase_diversity(data_frame):
    # Count the number of unique product groups for each customer
    purchase_diversity = data_frame.groupby('Customer_Id')['egroup'].nunique()

    # Merge purchase diversity back to the DataFrame
    data_frame['Purchase_Diversity'] = data_frame['Customer_Id'].map(purchase_diversity)

    return data_frame


def ohe(categorical_features,data):
   
   categorical_features = pd.get_dummies(data[categorical_features])
   print(type(categorical_features))
   return categorical_features


def map_master_sector(df): 
    
    sector_info = pd.read_excel(r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\main-data\sector_information.csv.xlsx')
    sector_info = sector_info.drop(['Suggestion'],axis = 1)
    sector_info['Sector'] = sector_info['Sector'].str.replace("\xa0","")

    mapped = pd.merge(df,sector_info,on="Sector",how='left')
    mapped = mapped.drop(['Sector'],axis = 1)

    return mapped

def create_training_data(): 
   
   data = get_data()
   data_backlog = data.loc[data['tran_type'] == 'Backlog']
   data = data.loc[data['tran_type'] == 'Shipment']

   date_columns = ["order_date", "efftive_date"]
   convert_to_datetime(date_columns,data)
   convert_to_float(data)
   data = data[data['order_date'] > '2018-12-12']
   numeric_columns =  ['qty', 'EUR']
   absolute_value(numeric_columns,data)

   data = calculate_purchase_variability(data)
   data = calculate_purchase_diversity(data)

   rfm_df = calculate_rfm(data)
   merged_df = merge_rfm(data,rfm_df)
   
   label_df = categorize_churn_status(data)
   labeled_df = pd.merge(merged_df,label_df,on='Customer_Id', how = 'left')
   final_df = group_by_customer(labeled_df)
   final_df = map_master_sector(final_df)
   
   opp = opportunity()
   final_df = pd.merge(final_df,opp,left_on='Customer_Id',right_on='erp_id',how = 'left')
   final_df['amount'] = np.where(final_df['close_date'] < final_df['order_date'], final_df['amount'], 0)
   
   bl = backlog(data_backlog)

   final_df = pd.merge(final_df,bl,left_on='Customer_Id',right_on='Customer_ID',how = 'left')
   final_df['Backlog'] = np.where(final_df['backlog_date'] < final_df['order_date'], final_df['Backlog'], 0)
   
   final_df.rename(columns={'amount' : 'Opportunity_Amt',
                            'Customer_Id_x' : 'Customer_Id'},inplace=True)
   final_df['Opportunity_Amt'].fillna(0,inplace=True)
   final_df['Backlog'].fillna(0,inplace=True)
   return final_df 
