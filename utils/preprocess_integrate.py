import os

from tqdm import tqdm
import numpy as np
import pandas as pd


all_orders = ['2019_Oct', '2019_Nov', '2019_Dec', '2020_Jan', '2020_Feb', '2020_Mar']
all_users = {}


if not os.path.isdir("data/"):
    os.mkdir("data/")


def clean_duplicates(temp, subdata, left=None, all_users=None):
    temp = temp[temp['user_id'].isin(all_users)]
    cols = [x for x in temp.columns.values if x not in ['user_id', 'event_time_utc']]
    
    if subdata == "value":
        temp[cols] = temp.groupby(["user_id", "event_time_utc"])[cols].transform("sum")
        temp = temp.fillna(0)

    elif subdata == "cov":
        temp = pd.merge(temp, left, on='user_id', how='left')
        temp[cols] = temp[cols].multiply(temp['views'], axis="index")
        temp[cols] = temp.groupby(["user_id", "event_time_utc"])[cols].transform("sum")
        temp = temp.fillna(0)
        temp[cols] = temp[cols].div(temp['views'], axis="index").replace(np.inf, 0)
        temp = temp.drop(['views'], axis=1)
    else:
        raise ValueError(f"Unknown subdata parameter: {subdata}")
    
    diff_users = set(all_users).difference(set(temp['user_id']))
    diff_users = pd.DataFrame(list(diff_users), columns=['user_id'])
    diff_users['event_time_utc'] = min(temp['event_time_utc'])
    
    for col in temp.columns.values:
        if col not in ['user_id', 'event_time_utc']:
            diff_users[col] = 0
    
    temp = pd.concat([temp, diff_users])
    temp = temp.drop_duplicates(subset=['user_id', 'event_time_utc'])
    return temp


for i in range(100):
    if os.path.exists(f'data/value_{i+1}.csv'):
        os.remove(f'data/value_{i+1}.csv')
    
    if os.path.exists(f'data/cov_{i+1}.csv'):
        os.remove(f'data/cov_{i+1}.csv')

for i, v in tqdm(enumerate(all_orders), "Wrap up Dataset"):
    for subdata in ["value", "cov"]:
        data = pd.read_csv(f'data/{subdata}_{v}.csv')
        data['event_time_utc'] = min(data['event_time_utc'])
        
        if subdata.startswith("value") and v == all_orders[0]:
            all_users = set(data['user_id'])
        
        for tm, df in data.groupby(by=['event_time_utc']):
            if os.path.exists(f"data/{subdata}_{str(tm)}.csv"):
                temp = pd.read_csv(f"data/{subdata}_{str(tm)}.csv")
                df = pd.concat([temp, df])
                
            if subdata == "value":
                df = clean_duplicates(df, subdata, None, all_users)
            else:
                left = pd.read_csv(f"data/value_{str(tm)}.csv")[['user_id', 'views']]
                df = clean_duplicates(df, subdata, left, all_users)

            df.to_csv(f"data/{subdata}_{str(tm)}.csv", header=True, index=None)
