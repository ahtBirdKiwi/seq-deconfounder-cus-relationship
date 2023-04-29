import os
import argparse

from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dataname")

args = parser.parse_args()
kwargs = vars(args)

dataname = kwargs['data']
chunksize = 30000

if not os.path.isfile("product_map.csv"):
    product_maps = {}
    category_maps = {}
else:
    product_maps = pd.read_csv("product_map.csv")
    product_maps = {k: v for k, v in zip(product_maps['product_id'], product_maps['category_id'])}
    category_maps = pd.read_csv("category_map.csv")
    category_maps = {k: v for k, v in zip(category_maps['category_id'], category_maps['category_code'])}


tmp = []
with pd.read_csv(dataname, chunksize=chunksize, iterator=True) as reader:
    for chunk in tqdm(reader, "processing_data..."):
        chunk = chunk.reset_index()
        product_cat = {k: v for k, v in zip(chunk['product_id'], chunk['category_id'])}
        category_cat = {k: v for k, v in zip(chunk['category_id'], chunk['category_code'])}
        product_maps.update(product_cat)
        category_maps.update(category_cat)
        chunk['event_time_utc'] = chunk['event_time'].str[:19]
        chunk['event_time_utc'] = pd.to_datetime(chunk['event_time_utc'], errors='coerce')
        chunk['event_time_utc'] = chunk['event_time_utc'] - pd.to_datetime("2019/10/01")
        chunk['event_time_utc'] = chunk['event_time_utc'].dt.days // 14 + 1
        chunk = chunk[['index', 'event_time_utc', 'event_type', 'product_id', 'brand', 'price', 'user_id']]
        tmp.append(chunk)
        
cleaned_data = pd.concat([x for x in tmp])
cleaned_data.to_csv("cleaned_" + dataname, header=True, index=False)
del cleaned_data

product_maps = pd.DataFrame.from_dict(product_maps, orient='index').reset_index()
product_maps.columns = ['product_id', 'category_id']
category_maps = pd.DataFrame.from_dict(category_maps, orient='index').reset_index()
category_maps.columns = ['category_id', 'category_code']

product_maps.to_csv("product_map.csv", header=True, index=False)
del product_maps
category_maps.to_csv("category_map.csv", header=True, index=False)
del category_maps
