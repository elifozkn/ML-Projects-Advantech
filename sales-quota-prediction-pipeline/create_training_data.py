import pandas as pd 
import numpy as np 
from connect_to_database import get_transaction_data,get_quota_data,get_historical_backlog,get_opportunity_data


def load_transaction():
    ### load and do simple preprocessing on transaction data
    transaction = get_transaction_data()
    transaction.rename({'Sales_id':'Sales_ID',
                        'Customer_Id':'Customer_ID',
                        'Sector':'sector','qty':
                        'Qty','tran_type':'Tran_Type'},axis=1,inplace=True)
    
    backlog = get_historical_backlog()
    
    columns_to_consider = ['Customer_ID', 'Tran_Type','Sales_ID','fact_entity','Qty', 'sector','EUR','order_date','efftive_date',
                       'order_no','Monetary','Frequency']
    transaction['order_date'] = pd.to_datetime(transaction['order_date'])
    transaction['efftive_date'] = pd.to_datetime(transaction['efftive_date'])
    transaction['Month_transaction']= transaction['order_date'].dt.month
    transaction['Year_transaction']= transaction['order_date'].dt.year
    transaction['EUR'] = transaction['EUR'].abs()
    transaction['Month_efftive']= transaction['efftive_date'].dt.month
    transaction['Year_efftive']= transaction['efftive_date'].dt.year

    
    backlog['efftive_date'] = pd.to_datetime(backlog['efftive_date'])
    backlog['Month_efftive'] = backlog['efftive_date'].dt.month
    backlog['Year_efftive'] = backlog['efftive_date'].dt.year
    backlog['Sales_ID'] = backlog['Sales_ID'].astype('int64')
    transaction = transaction.loc[transaction['Year_efftive'] > 2018]

    backlog = backlog.groupby(['Sales_ID','Year_efftive','Month_efftive']).agg({'EUR':'sum'}).reset_index()
    backlog.rename(columns={'EUR': 'Backlog'}, inplace=True)
    freq = transaction.groupby(['Sales_ID','Year_efftive','Month_efftive']).agg({'efftive_date' : 'count'}).reset_index()
    freq.rename(columns={'efftive_date': 'Frequency'}, inplace=True)

    mon = transaction.groupby(['Sales_ID','Year_efftive','Month_efftive']).agg({'EUR' : 'sum'}).reset_index()
    mon.rename(columns={'EUR': 'Monetary'}, inplace=True)
    
    t = pd.merge(freq, mon, on = ["Sales_ID",'Year_efftive','Month_efftive'])
    t['Monetary'] = t['Monetary'].astype(float)
    t['Frequency'] = pd.qcut(t['Frequency'], q=5, labels=False)
    t['Monetary'] = pd.qcut(t['Monetary'], q=5, labels=False)

    transaction = pd.merge(transaction, t, on = ["Sales_ID",'Year_efftive','Month_efftive'])
    transaction = transaction[transaction['Sales_ID'] != 'GPSales']
    transaction['Sales_ID'] = transaction['Sales_ID'].astype('int64')
    transaction = pd.merge(transaction, backlog, on = ["Sales_ID", 'Year_efftive', 'Month_efftive'], how='left')
    transaction_shipment = transaction.loc[transaction['Tran_Type']=='Shipment']
    transaction = transaction[columns_to_consider]
    return transaction,transaction_shipment


def load_sales_quota(): 
    # load and do simple preprocessing on sales quota data
    quota = get_quota_data()
    columns_to_consider_sales_quota = ['Sector', 'Amount','sales_id','SalesName','Fact_Entity','year_id','Month','efftive_date','Master_Sector']
    quota = quota[columns_to_consider_sales_quota]

    quota['Amount'] = quota['Amount'] * 1000
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
             '1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12}
    quota['Month'] = quota['Month'].map(month_map)
    return quota 

def get_spread(df):
    grouped  = df.groupby('sales_id').agg(mean_amount=('Amount', 'mean'),
                                                         std_amount=('Amount', 'std')).reset_index()
    grouped = grouped.rename(columns = {'std_amount' : 'Spread',
                                        'mean_amount':'Historical_Mean'})
    spread_df = pd.merge(df,grouped,on='sales_id',how = 'left')
    return spread_df


def get_opporunity(quota):
    opportunity = get_opportunity_data()
    
    columns_to_consider_opportunity = ['probability',
                                   'won_lost_Reason','sales_code_text','amount','estimated_amount',
                                   'account_status','close_date','contract_date',
                                   'created_date','stage']

    opportunity = opportunity[columns_to_consider_opportunity]

    opportunity['sales_code_text'] = pd.to_numeric(opportunity['sales_code_text'], errors='coerce')
    opportunity['created_date'] = pd.to_datetime(opportunity['created_date'])
    opportunity['year_opp'] = opportunity['created_date'].dt.year
    opportunity['month_opp'] = opportunity['created_date'].dt.month

    opportunity.dropna(subset=['sales_code_text'], inplace=True)
    opportunity['sales_code_text'] = opportunity['sales_code_text'].astype(int)
    opportunity_quota = pd.merge(quota,opportunity,left_on='sales_id',right_on='sales_code_text',how='left')
    opportunity_quota = opportunity_quota.loc[(opportunity_quota['year_id'] == opportunity_quota['year_opp']) & 
                                          (opportunity_quota['Month'] == opportunity_quota['month_opp'])]
    return opportunity_quota



def calculate_win_rate(prob_list):
    total_leads = len(prob_list)
    won_leads = prob_list.count(100.0)
    return (won_leads / total_leads) * 100 if total_leads > 0 else 0

def calculate_partial_win_rate(prob_list):
    total_leads = len(prob_list)
    won_leads = sum(1 for prob in prob_list if prob >= 75)
    return (won_leads / total_leads) * 100 if total_leads > 0 else 0

def calculate_opportunity_stats_sales_wise(quota):
    opportunity_quota = get_opporunity(quota)
    w = opportunity_quota.groupby(['sales_id','year_id','Month'])['probability'].apply(list).reset_index()
    w['Win_Rate'] = w['probability'].apply(calculate_win_rate)
    w['Partial_Win_Rate'] = w['probability'].apply(calculate_partial_win_rate)  

    return w

def get_lag_sales_id(df): 
    dfs = []
    for sales_id in df['sales_id'].unique(): 

        ac = df.copy()
        ac = ac[ac['sales_id'] == sales_id] 
        ac.set_index(['year_id', 'Month'], inplace=True)
        idx = pd.MultiIndex.from_product([ac.index.levels[0], range(1, 13)], names=['year_id', 'Month'])
        ac = ac.reindex(idx)
        ac.reset_index(inplace=True)

        # Sort DataFrame based on year_id and Month
        ac.sort_values(by=['year_id', 'Month'], inplace=True)
        ac.loc[ac['year_id'] == 2024, 'sales_id'] = sales_id
        # Perform lag operation
        ac['Lagged_EUR'] = ac.groupby('sales_id')['EUR'].shift(12)

        ac['Lagged_Quota'] = ac.groupby('sales_id')['Amount'].shift(12)
        ac['Lagged_Monetary'] = ac.groupby('sales_id')['Monetary'].shift(12)
        ac['Lagged_Frequency'] = ac.groupby('sales_id')['Frequency'].shift(12)
        #ac['Lagged_Partial_Win_Rate'] = ac.groupby('sales_id')['Partial_Win_Rate'].shift(12)
        dfs.append(ac)

    return dfs

def remove_outliers(df, column_name, multiplier=1.5):

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return filtered_df


def prepare(): 
    print('Preparing data...')
    backlog,df = load_transaction()
    quota = load_sales_quota()
    df = df.groupby(['Sales_ID','Year_efftive','Month_efftive']).agg({'EUR': 'sum',
                                                                 'Monetary' : 'first',
                                                                 'Frequency' : 'first',
                                                                 'Backlog':'sum'}).reset_index()
    quota = quota.groupby(['sales_id','year_id','Month']).agg({'Amount': 'sum',
                                                           'Master_Sector' : 'first',
                                                           'Fact_Entity' : 'first',
                                                           }).reset_index()
    transaction_quota = pd.merge(df,quota,left_on=['Sales_ID','Year_efftive','Month_efftive'],right_on=['sales_id','year_id','Month'],how='right')
    transaction_quota = transaction_quota.drop(['Sales_ID','Year_efftive','Month_efftive'],axis = 1)

    transaction_quota['EUR'] = transaction_quota.groupby('sales_id')['EUR'].transform(lambda x: x.fillna(x.mean()))
    transaction_quota['Monetary'] = transaction_quota.groupby('sales_id')['Monetary'].transform(lambda x: x.fillna(x.mean()))
    transaction_quota['Frequency'] = transaction_quota.groupby('sales_id')['Frequency'].transform(lambda x: x.fillna(x.mean()))
    opportunity = calculate_opportunity_stats_sales_wise(quota)
    transaction_quota = pd.merge(transaction_quota, opportunity, on=['sales_id', 'year_id', 'Month'], how='left')


    ls = get_lag_sales_id(transaction_quota)
    combined_df = pd.concat(ls,ignore_index=True)
    combined_df = get_spread(combined_df)
    opportunity = calculate_opportunity_stats_sales_wise(quota)
    combined_df = pd.merge(combined_df, opportunity, on=['sales_id', 'year_id', 'Month'], how='left')

    combined_df.fillna(0, inplace=True)

    one_hot_encoded_column = pd.get_dummies(combined_df['Master_Sector'])

    combined_df = pd.concat([combined_df, one_hot_encoded_column], axis=1)
    combined_df.drop(columns = ['Master_Sector','EUR','Fact_Entity'],axis = 1, inplace = True)
    columns_to_standardize = ['Monetary','Spread','Backlog','Historical_Mean' ,'Frequency' , 'Lagged_EUR' , 'Lagged_Quota', 'Lagged_Monetary','Lagged_Frequency']
    for column in columns_to_standardize:
        combined_df[f'{column}_std'] = (combined_df[column] - combined_df[column].min()) /(combined_df[column].max() - combined_df[column].min())

    return combined_df

def add_backlog(combined_df):
    
    transaction = pd.read_csv(r'C:\Users\elif.yozkan\Desktop\advantech-sales-quota-prediction\future_backlog.csv')
    transaction['efftive_date'] = pd.to_datetime(transaction['efftive_date'])
    transaction['EUR'] = transaction['EUR'].abs()
    transaction['Month']= transaction['efftive_date'].dt.month
    transaction['year_id']= transaction['efftive_date'].dt.year
    transaction = transaction[transaction['Tran_Type'] == 'Backlog']
    transaction = transaction.loc[transaction['year_id'] > 2018]
    transaction = transaction[['Month','year_id','EUR','Sales_ID','efftive_date']]
    transaction.rename({'Sales_ID':'sales_id','EUR':'Backlog'},axis=1,inplace=True)
    transaction = transaction.groupby(['sales_id','year_id','Month']).agg({'Backlog' : 'sum'}).reset_index()
    combined_df = pd.merge(combined_df,transaction,on = ['sales_id','year_id','Month'],how = 'left')
    return combined_df


def calculate_cagr():
    # Ensure the dataframe is sorted by Sales_ID and Year_efftive
    b,df = load_transaction()
    df = df.groupby(['Year_efftive','Sales_ID']).agg({'EUR':'sum'}).reset_index()

    df = df.sort_values(by=['Sales_ID', 'Year_efftive'])
    df['EUR'] = df['EUR'].astype('float64')
    # Create an empty list to store the results
    cagr_list = []

    # Get the unique Sales_IDs
    sales_ids = df['Sales_ID'].unique()

    # Loop through each Sales_ID
    for sales_id in sales_ids:
        # Filter the dataframe for that specific Sales_ID
        sales_data = df[df['Sales_ID'] == sales_id]
        
        # Ensure there are at least two distinct years to compute CAGR
        if len(sales_data) > 1:
            # Get the initial and final revenue values
            initial_year = sales_data.iloc[0]['Year_efftive']
            final_year = sales_data.iloc[-1]['Year_efftive']
            initial_value = sales_data.iloc[0]['EUR']
            final_value = sales_data.iloc[-1]['EUR']
            
            # Calculate the number of years
            n = final_year - initial_year
            # Compute CAGR
            if initial_value > 0 and n > 0:
                cagr = (final_value / initial_value) ** (1 / n) - 1
            else:
                cagr = 0  # Handle cases where initial value is zero or if there's only one year

            # Append result to the list
            cagr_list.append({'Sales_ID': sales_id, 'CAGR': cagr})

    # Convert the list of dictionaries to a DataFrame
    cagr_df = pd.DataFrame(cagr_list)
    return cagr_df

def categorize_growth_potential(cagr_df):
    # Define quantiles for categorization based on CAGR
    low_threshold = cagr_df['CAGR'].quantile(0.25)
    high_threshold = cagr_df['CAGR'].quantile(0.50)

    # Define a function to assign growth potential based on CAGR
    def growth_potential_category(cagr):
        if cagr >= high_threshold:
            return 'High'
        elif cagr >= low_threshold:
            return 'Medium'
        else:
            return 'Low'

    # Apply the categorization function to each row in the CAGR dataframe
    cagr_df['Growth_Potential'] = cagr_df['CAGR'].apply(growth_potential_category)
    print(cagr_df.columns)
    return cagr_df

def prepare_inference_data():
    print('Preparing inference data...')

    backlog,df = load_transaction()
    quota = load_sales_quota()
    df = df.groupby(['Sales_ID','Year_efftive','Month_efftive']).agg({'EUR': 'sum',
                                                                 'Monetary' : 'first',
                                                                 'Frequency' : 'first',
                                                                 'Backlog':'sum'}).reset_index()
    quota = quota.groupby(['sales_id','year_id','Month']).agg({'Amount': 'sum',
                                                           'Master_Sector' : 'first',
                                                           'Fact_Entity' : 'first',
                                                           }).reset_index()
    transaction_quota = pd.merge(df,quota,left_on=['Sales_ID','Year_efftive','Month_efftive'],right_on=['sales_id','year_id','Month'],how='right')
    transaction_quota = transaction_quota.drop(['Sales_ID','Year_efftive','Month_efftive'],axis = 1)

    transaction_quota['EUR'] = transaction_quota.groupby('sales_id')['EUR'].transform(lambda x: x.fillna(x.mean()))
    transaction_quota['Monetary'] = transaction_quota.groupby('sales_id')['Monetary'].transform(lambda x: x.fillna(x.mean()))
    transaction_quota['Frequency'] = transaction_quota.groupby('sales_id')['Frequency'].transform(lambda x: x.fillna(x.mean()))

    ls = get_lag_sales_id(transaction_quota)
    temp_df = pd.concat(ls,ignore_index=True)
    temp_df = get_spread(temp_df)
    opportunity = calculate_opportunity_stats_sales_wise(quota)
    #opportunity = get_opporunity(quota)
    combined_df = pd.merge(temp_df, opportunity, on=['sales_id', 'year_id', 'Month'], how='left')
    cagr_df = calculate_cagr()
    cagr_df = categorize_growth_potential(cagr_df=cagr_df)
    combined_df = pd.merge(combined_df,cagr_df,left_on='sales_id',right_on='Sales_ID',how = 'left')
    print(combined_df.columns)
    combined_df['Growth_Potential'].fillna('Low',inplace=True)
    combined_df.fillna(0, inplace=True)

    combined_df = combined_df[['EUR','Amount','Monetary','Spread','Frequency','Historical_Mean','Partial_Win_Rate','Master_Sector','sales_id','year_id','Month','Fact_Entity','Growth_Potential']]
    one_hot_encoded_column = pd.get_dummies(combined_df['Master_Sector'])

    combined_df = pd.concat([combined_df, one_hot_encoded_column], axis=1)
    combined_df = combined_df.loc[combined_df['year_id'] == combined_df['year_id'].max()]
    combined_df['year_id'].replace(2024,2025,inplace=True)

    combined_df = add_backlog(combined_df)
    columns_to_standardize = ['Monetary','Spread','Historical_Mean' ,'Frequency' , 'EUR','Amount','Backlog']
    for column in columns_to_standardize:
        combined_df[f'{column}_std'] = (combined_df[column] - combined_df[column].min()) /(combined_df[column].max() - combined_df[column].min())

    combined_df.rename({'EUR_std':'Lagged_EUR_std','Monetary_std':'Lagged_Monetary_std','Frequency_std':'Lagged_Frequency_std','Amount_std':'Lagged_Quota_std'},axis =1,inplace = True)
    combined_df = combined_df[['Lagged_EUR_std','Spread_std','Backlog_std','Growth_Potential',
                                      'Lagged_Quota_std','Lagged_Monetary_std','Lagged_Frequency_std','Fact_Entity','Master_Sector','EUR','Amount',
                                      'Month','CIOT', 'EIoT', 'IIoT', 'SIOT','year_id','sales_id']]
    combined_df.fillna(0,inplace=True)
    return combined_df


def prepare_inference_data1():
    print('Preparing inference data...')

    backlog, df = load_transaction()
    quota = load_sales_quota()
    
    df = df.groupby(['Sales_ID', 'Year_efftive', 'Month_efftive']).agg({
        'EUR': 'sum',
        'Monetary': 'first',
        'Frequency': 'first',
        'Backlog': 'sum'
    }).reset_index()
    
    quota = quota.groupby(['sales_id', 'year_id', 'Month']).agg({
        'Amount': 'sum',
        'Master_Sector': 'first',
        'Fact_Entity': 'first',
        'SalesName' : 'last'
    }).reset_index()
    
    transaction_quota = pd.merge(df, quota, left_on=['Sales_ID', 'Year_efftive', 'Month_efftive'], right_on=['sales_id', 'year_id', 'Month'], how='right')
    transaction_quota = transaction_quota.drop(['Sales_ID', 'Year_efftive', 'Month_efftive'], axis=1)

    # Fill EUR based on the month condition
    def fill_eur(row):
        if row['Month'] < 8:
            return 0 if pd.isna(row['EUR']) else row['EUR']
        else:
            return np.nan  # Keep as NaN for interpolation later
    
    # Apply the filling condition for months < 8
    transaction_quota['EUR'] = transaction_quota.apply(fill_eur, axis=1)

    # Apply sinusoidal interpolation for months >= 8
    def sinusoidal_interpolation(x):
        n = len(x)
        time = np.linspace(0, 2 * np.pi, n)
        interp_values = np.interp(time, time[~np.isnan(x)], x[~np.isnan(x)])
        return interp_values
    transaction_quota['EUR'] = transaction_quota['EUR'].astype('float64')
    print(type(transaction_quota['Month'][0]))
    transaction_quota.loc[transaction_quota['Month'] >= 8, 'EUR'] = transaction_quota.groupby('sales_id')['EUR'].transform(
        lambda x: sinusoidal_interpolation(x) if x.isna().any() else x)

    # Fill other missing values for Monetary and Frequency
    transaction_quota['Monetary'] = transaction_quota.groupby('sales_id')['Monetary'].transform(lambda x: x.fillna(x.mean()))
    transaction_quota['Frequency'] = transaction_quota.groupby('sales_id')['Frequency'].transform(lambda x: x.fillna(x.mean()))

    # The rest of your original code follows...
    ls = get_lag_sales_id(transaction_quota)
    temp_df = pd.concat(ls, ignore_index=True)
    temp_df = get_spread(temp_df)
    opportunity = calculate_opportunity_stats_sales_wise(quota)
    
    combined_df = pd.merge(temp_df, opportunity, on=['sales_id', 'year_id', 'Month'], how='left')
    cagr_df = calculate_cagr()
    cagr_df = categorize_growth_potential(cagr_df=cagr_df)
    
    combined_df = pd.merge(combined_df, cagr_df, left_on='sales_id', right_on='Sales_ID', how='left')
    combined_df['Growth_Potential'].fillna('Low', inplace=True)
    combined_df.fillna(0, inplace=True)
    
    combined_df = combined_df[['EUR', 'Amount', 'Monetary', 'Spread', 'Frequency', 'Historical_Mean', 'Partial_Win_Rate', 'Master_Sector', 'sales_id', 'year_id', 'Month', 'Fact_Entity', 'Growth_Potential','CAGR','SalesName']]
    one_hot_encoded_column = pd.get_dummies(combined_df['Master_Sector'])
    
    combined_df = pd.concat([combined_df, one_hot_encoded_column], axis=1)
    combined_df = combined_df.loc[combined_df['year_id'] == combined_df['year_id'].max()]
    combined_df['year_id'].replace(combined_df['year_id'].max(), combined_df['year_id'].max()+1, inplace=True)
    
    combined_df = add_backlog(combined_df)
    columns_to_standardize = ['Monetary', 'Spread', 'Historical_Mean', 'Frequency', 'EUR', 'Amount', 'Backlog']
    for column in columns_to_standardize:
        combined_df[f'{column}_std'] = (combined_df[column] - combined_df[column].min()) / (combined_df[column].max() - combined_df[column].min())

    combined_df.rename({'EUR_std': 'Lagged_EUR_std', 'Monetary_std': 'Lagged_Monetary_std', 'Frequency_std': 'Lagged_Frequency_std', 'Amount_std': 'Lagged_Quota_std'}, axis=1, inplace=True)
    
    combined_df = combined_df[['Lagged_EUR_std', 'CAGR','Backlog_std', 'Growth_Potential', 'Lagged_Quota_std', 'Lagged_Monetary_std', 'Lagged_Frequency_std', 'Fact_Entity', 'Master_Sector', 'EUR', 'Amount', 'Month', 'CIOT', 'EIoT', 'IIoT', 'SIOT', 'year_id', 'sales_id','SalesName']]
    combined_df.fillna(0, inplace=True)
    
    return combined_df


