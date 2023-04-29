import os
import argparse

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dataname")

args = parser.parse_args()
kwargs = vars(args)

dataname = kwargs['data']
outdataname = dataname.replace('cleaned_', '').replace('.csv', '')


def build_cov(data):
    covar = data[(data['event_type'] == 'view')].groupby(by=['user_id', 'event_time_utc', 'product_id'])['price'].count().reset_index()
    covar.columns = ['user_id', 'event_time_utc', 'category', 'count']
    covar = pd.pivot_table(data=covar, values='count', index=['user_id', 'event_time_utc'], columns='category', fill_value=0)
    covar = covar.div(covar.sum(axis=1), axis=0).reset_index()

    covar.to_csv(f"data/cov_{outdataname}.csv", header=True, index=False)


def build_vals(data):
    all_users = np.unique(data['user_id'])

    values = data[(data['event_type'] == 'purchase')].groupby(by=['user_id', 'event_time_utc'])['price'].sum().reset_index()
    values.columns = ['user_id', 'event_time_utc', 'value']
    
    for wks, subdf in values.groupby(by='event_time_utc'):
        zero_value = pd.DataFrame(columns=['user_id', 'event_time_utc', 'value'])
        zero_value['user_id'] = list(set(all_users).difference(set(subdf['user_id'])))
        zero_value['event_time_utc'] = wks
        zero_value['value'] = 0
        values = pd.concat([values, zero_value])
    del zero_value

    perf = data[(data['event_type'] == 'view')].groupby(by=['user_id', 'event_time_utc'])['index'].count().reset_index()
    perf.columns = ['user_id', 'event_time_utc', 'views']

    values = pd.merge(values, perf, on=['user_id', 'event_time_utc'], how='left').fillna(0)

    cart = data[(data['event_type'] == 'cart')].groupby(by=['user_id', 'event_time_utc'])['index'].count().reset_index()
    cart.columns = ['user_id', 'event_time_utc', 'cart']

    values = pd.merge(values, cart, on=['user_id', 'event_time_utc'], how='left').fillna(0)
    values.to_csv(f"data/value_{outdataname}.csv", header=True, index=False)


if __name__ == "__main__":
    
    
    if not os.path.isfile("user_list.csv"):
        user_list = {}
    else:
        user_list = pd.read_csv("user_list.csv")
        user_list = {k: v for k, v in zip(user_list['user_id'], user_list['earliest_month'])}

    product_maps = pd.read_csv("product_map.csv")
    product_maps = {k: v for k, v in zip(product_maps['product_id'], product_maps['category_id'])}
    category_maps = pd.read_csv("category_map.csv")
    category_maps = {k: v for k, v in zip(category_maps['category_id'], category_maps['category_code'])}

    data = pd.read_csv(dataname)
    # data['event_time_utc'] = [min(2, x) for x in data['event_time_utc']]
    data['product_id'] = [str(category_maps[product_maps[x]]).split('.')[0] for x in data['product_id']]
    
    build_cov(data)
    build_vals(data)
