import pandas as pd 


def arrange_customer_type(df):
    # Create a list of the one-hot encoded columns
    one_hot_columns = ['Customer_AOL', 'Customer_CP', 'Customer_KA']

    for column in one_hot_columns: 
        df[column] = df[column].astype(int)
        
    # Create a new column 'customer_type' by combining the one-hot columns
    df['customer_type'] = df[one_hot_columns].idxmax(axis=1).str.replace('CustomerType', '')
    df.drop(one_hot_columns, axis=1, inplace=True)
    return df

def denormalize_qty(df): 
    # Define mean and standard deviation of the original 'qty' column
    mean_qty = df['qty'].mean()
    std_qty = df['qty'].std()

    # Apply inverse Z-score normalization to convert the values back to the original scale
    df['qty'] = (df['qty'] * std_qty) + mean_qty

    return df 


def post_process(df): 

    df = arrange_customer_type(df)
    df = denormalize_qty(df)

    return df 