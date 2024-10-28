import pyodbc
import pandas as pd
import sqlalchemy
import urllib

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

    return df

def get_to_be_predicted_data():
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
    sql_query = 'SELECT [Frequency_Score],[Monetary_Score],[RFM_Score],[qty],[cust_country],[Label],[Customer_Id],[Customer_AOL],[Customer_CP],[Customer_KA] FROM customer_retention_to_be_predicted'

    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
    finally:
        cursor.close()
    
    column_names = ["Frequency_Score","Monetary_Score","RFM_Score","qty","cust_country","Label","Customer_Id","Customer_AOL","Customer_CP","Customer_KA"]

    data_list = [dict(zip(column_names, row)) for row in data]

    df = pd.DataFrame(data_list)

    conn.close()

    return df

def get_opportunity_data():
    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model' #add username
    password = '@Advantech!532' #add password

    connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}'

    try:
        conn = pyodbc.connect(connection_string)
        print('Retrieving Quota Data...')
    except Exception as e:
        print(f"Error connecting to SQL Server: {str(e)}")


    sql_query = 'SELECT [opportunity_name],[oppority_record_type],[industry],[industry_other],[application],[application_other],[account_name],[erp_id],[account_legacy_system_id],[account_country],[account_status],[parent_account],[type],[stage],[probability],[won_lost_Reason],[sales_code],[opportunity_owner],[opportunity_currency],[close_date],[contract_date],[first_shipping_date],[estimated_amount],[amount],[revenue_current_year],[end_customer],[design_win_status],[primary_campaign_source],[fao_source],[created_by],[created_date],[rbu_text],[sector_text],[sales_team_text],[sales_code_text],[opportunity_legacy_system_id],[mailbee_trace_id],[city_country],[project_class],[owner_business_group__c],[product_customization_description__c],[wise_iot_tag__c],[product_customization_needed__c],[assign_to_partner__c],[registered_by_partner__c],[converted_lead_source__c],[territory_text__c],[ticketing_num__c],[parent_rbu__c],[position_id__c],[opportunity_id] FROM [SALES].[dbo].[store_aeu_opportunity]'

    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
    finally:
        cursor.close()
    
    column_names = ['opportunity_name','oppority_record_type','industry','industry_other','application','application_other','account_name','erp_id','account_legacy_system_id','account_country','account_status','parent_account','type','stage','probability','won_lost_Reason','sales_code','opportunity_owner','opportunity_currency','close_date','contract_date','first_shipping_date','estimated_amount','amount','revenue_current_year','end_customer','design_win_status','primary_campaign_source','fao_source','created_by','created_date','rbu_text','sector_text','sales_team_text','sales_code_text','opportunity_legacy_system_id','mailbee_trace_id','city_country','project_class','owner_business_group__c','product_customization_description__c','wise_iot_tag__c','product_customization_needed__c','assign_to_partner__c','registered_by_partner__c','converted_lead_source__c','territory_text__c','ticketing_num__c','parent_rbu__c','position_id__c','opportunity_id']

    data_list = [dict(zip(column_names, row)) for row in data]

    df = pd.DataFrame(data_list)

    conn.close()

    return df

def get_historical_backlog():
    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model' #add username
    password = '@Advantech!532' #add password

    connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}'

    try:
        conn = pyodbc.connect(connection_string)
        print('Retrieving Historical Backlog...')
    except Exception as e:
        print(f"Error connecting to SQL Server: {str(e)}")


    sql_query = 'SELECT TOP (1000) [Customer_ID],[Tran_Type],[Sales_ID],[fact_entity],[Qty],[sector],[EUR],[order_date],[efftive_date],[order_no] FROM [SALES].[dbo].[Backlog_History]'

    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
    finally:
        cursor.close()
    
    column_names = ["Customer_ID","Tran_Type","Sales_ID","fact_entity","Qty","sector","EUR","order_date","efftive_date","order_no"]

    data_list = [dict(zip(column_names, row)) for row in data]

    df = pd.DataFrame(data_list)

    conn.close()

    return df

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

def write_to_sql(df,table_name,overwrite = True):
    if overwrite:
        clear_table(table_name)
    
    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model'
    password = '@Advantech!532'
    driver = 'ODBC Driver 17 for SQL Server'  

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server_name};"
        f"DATABASE={database_name};"
        f"UID={username};"
        f"PWD={password};"
    )

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    columns = ', '.join(df.columns)
    placeholders = ', '.join(['?'] * len(df.columns))
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    for row in df.itertuples(index=False):
        cursor.execute(insert_query, row)

    conn.commit()
    cursor.close()
    conn.close()



def clear_table(table_name):

    server_name = '172.21.34.41'
    database_name = 'SALES'
    username = 'ai.model'
    password = '@Advantech!532'
    driver = 'ODBC Driver 17 for SQL Server'
    
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server_name};"
        f"DATABASE={database_name};"
        f"UID={username};"
        f"PWD={password};"
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    clear_query = f"TRUNCATE TABLE {table_name}"

    cursor.execute(clear_query)
    conn.commit()
    cursor.close()
    conn.close()


