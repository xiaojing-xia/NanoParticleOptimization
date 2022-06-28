import gspread
import pandas as pd

def get_ws_gspread(cred_filename, sheet_name, worksheet_name = 'Sheet1'):
    cred = gspread.service_account(cred_filename)
    sheet = cred.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    
    return worksheet


def get_df_gspread(cred_filename, sheet_name, worksheet_name = 'Sheet1'):
    worksheet = get_ws_gspread(cred_filename, sheet_name, worksheet_name)
    rows = worksheet.get_all_values()
    df = pd.DataFrame.from_records(rows)
    df = df.rename(columns = df.iloc[0]).drop(df.index[0]).astype(float)
    
    return df