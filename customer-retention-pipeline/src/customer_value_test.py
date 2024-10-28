from connect_to_database import get_training_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split,KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import pyodbc
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, r2_score,mean_absolute_error
def get_training_data():
    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model' #add username
    password = '@Advantech!532' #password

    connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}'

    try:
        conn = pyodbc.connect(connection_string)
        print('Connected to the Database')
    except Exception as e:
        print(f"Error connecting to SQL Server: {str(e)}")
    sql_query = 'SELECT [Frequency_Score],[Monetary_Score],[RFM_Score],[cust_country],[Customer_Id],[Customer_AOL],[Customer_CP],[Customer_KA],[target_encoded],[qty],[Label] FROM customer_retention_training_data'

    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
    finally:
        cursor.close()
    
    column_names = ["Frequency_Score","Monetary_Score","RFM_Score","cust_country","Customer_Id","Customer_AOL","Customer_CP","Customer_KA","target_encoded","qty","Label"]

    data_list = [dict(zip(column_names, row)) for row in data]

    df = pd.DataFrame(data_list)

    conn.close()

    return df


def get_data():
    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model' #add username
    password = '@Advantech!532' #add password

    connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}'

    try:
        conn = pyodbc.connect(connection_string)
        print('Connected to the Database')
    except Exception as e:
        print(f"Error connecting to SQL Server: {str(e)}")


    sql_query = 'SELECT [tran_type],[fact_1234],[bomseq],[dec01],[gameFLAG],[fact_zone],[fact_entity],[Sector],[Sales_id],[Customer_Id],[egroup],[edivision],[acl_pdl],[item_no],[efftive_date],[sale_type],[breakdown],[itp_find],[qty],[cancel_flag],[us_amt],[TWD],[EUR],[RMB],[JPY],[SGD],[AUD],[MYR],[KRW],[BRL],[THB],[BUsector],[record_id],[tr_entity],[Ship_Site],[cust_country],[fact_country],[us_cost],[us_ex_rate],[order_no],[order_date],[idh_inv_nbr],[sam_sw],[EndCust],[splitFLAG],[Site_ID],[WebAccess],[PGI_Date],[FirstDate],[YesNo],[alongWO],[m_type],[VM2],[tr_line],[PO] FROM iv_eai_acldw_g_sale_fact_bomexpand'

    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
    finally:
        cursor.close()
    
    column_names = ["tran_type","fact_1234","bomseq","dec01","gameFLAG","fact_zone","fact_entity","Sector","Sales_id",
                    "Customer_Id",'egroup','edivision','acl_pdl','item_no','efftive_date','sale_type',
                    'breakdown','itp_find','qty','cancel_flag','us_amt','TWD','EUR','RMB','JPY','SGD','AUD','MYR',
                    'KRW','BRL','THB','BUsector','record_id','tr_entity','Ship_Site','cust_country','fact_country',
                    'us_cost','us_ex_rate','order_no','order_date','sales_site_id','idh_inv_nbr','sam_sw',
                    'EndCust','splitFLAG','Site_ID','WebAccess','PGI_Date','FirstDate','YesNo','alongWO',
                    'FromCost','m_type','VM2','tr_line','PO']

    data_list = [dict(zip(column_names, row)) for row in data]

    df = pd.DataFrame(data_list)
    conn.close()
    df = df[['Customer_Id', 'EUR']]
    df = df.groupby('Customer_Id').agg({'EUR': 'sum'}).reset_index()
    print(df.head())
    return df


    from sklearn.ensemble import RandomForestRegressor


def train_randomforest(df):
    df = df.loc[df['EUR'] > 0].reset_index(drop=True)
    X = df.drop(['Label', 'EUR', 'Customer_Id', 'cust_country'], axis=1)
    X['Customer_AOL'] = X['Customer_AOL'].astype(int)
    X['Customer_CP'] = X['Customer_CP'].astype(int)
    X['Customer_KA'] = X['Customer_KA'].astype(int)
    y = df['EUR']

    # Initialize the RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform stratified 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_list = []
    mape_list = []
    r2_list = []
    mae_list = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = pd.DataFrame(X).iloc[train_idx], pd.DataFrame(X).iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_test = y_test.apply(float)
        # Fit the RandomForestRegressor
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_regressor.predict(X_test)
    

        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mape = np.abs((y_test - y_pred) / (y_test))*100
        r2 = r2_score(y_test, y_pred)
        mae =  mean_absolute_error(y_test, y_pred)

        print(f"MAPE : {np.mean(mape)}")

        mse_list.append(mse)
        r2_list.append(r2)
        mae_list.append(mae)

    # Calculate the average regression metrics
    avg_mse = np.mean(mse_list)
    avg_r2 = np.mean(r2_list)
    avg_mae = np.mean(mae_list)

    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    # Save the trained model to a file
    model_filename = r'C:\Users\elif.yozkan\Desktop\customer-retention-pipeline\src\models\randomforest.model'
    return model_filename, y_pred

def train():
    df = get_training_data()
    rev = get_data()
    df = df.merge(rev, on='Customer_Id')
    model_filename, y_pred = train_randomforest(df)
train()