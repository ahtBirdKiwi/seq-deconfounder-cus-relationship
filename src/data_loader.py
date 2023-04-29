import logging

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class IterativeLoader():
    def __init__(self, random_sample=700000, use_hidden=True):
        self.cnt = 1
        self.random_sample = random_sample
        self.all_users = []
        self.use_hidden = use_hidden

    def __getitem__(self, index):
        if os.path.exists(f"data/value_{self.cnt}.csv"):
            val_data = pd.read_csv(f"data/value_{self.cnt}.csv")
            cov_data = pd.read_csv(f"data/cov_{self.cnt}.csv")
            self.cnt += 1
            return val_data, cov_data
        return None, None

    def get_init_hidden_data(self, sample=700000):
        # val_data_ = pd.read_csv("data/value_7.csv")
        val_data_ = pd.read_csv("data/value_6.csv")
        
        user_num = len(np.unique(val_data_['user_id']))
        
        if user_num > sample:
            users = val_data_['user_id'].sample(sample)
        else:
            users = val_data_['user_id']
        
        del val_data_
        
        val_data = pd.read_csv("data/value_1.csv")
        cov_data = pd.read_csv("data/cov_1.csv")
        val_data = val_data[val_data['user_id'].isin(users)]
        cov_data = cov_data[cov_data['user_id'].isin(users)]
        
        return val_data, cov_data
    
    def get_data(self, index):
        val_data = pd.read_csv(f"data/value_{index}.csv")
        cov_data = pd.read_csv(f"data/cov_{index}.csv")
        
        if self.use_hidden and os.path.exists("data/hidden_states.h5"):
            print("Use Hidden States")
            hidden = self.get_hidden_states()
            users_in_hidden = hidden['user_id']
            val_data = val_data[val_data['user_id'].isin(users_in_hidden)]
            cov_data = cov_data[cov_data['user_id'].isin(users_in_hidden)]
            return val_data, cov_data
        
        return val_data, cov_data


    def get_hidden_states(self):
        states_data = pd.read_hdf("data/hidden_states.h5")
        return states_data


    def get_assignment_params(self):
        if not os.path.exists("data/assignment_params.h5"):
            proceed = input("Initialize assignment parameters? (Y/N)")
            
            if proceed == "Y":
                val_data = self.get_data(1)[0]
                users = self.get_hidden_states()
                users = users['user_id']
                val_data = val_data[val_data['user_id'].isin(users)]
                assignment_data = pd.DataFrame(columns=["user_id", "params_1"])
                assignment_data['user_id'] = val_data['user_id']
                # mu, vars = np.mean(val_data['value']), np.var(val_data['value'])
                # alpha = mu * (mu * (1 - mu) / vars - 1)
                # beta = (1 - mu) * (mu * (1 - mu) / vars - 1)
                assignment_data["params_1"] = [[x, y] for x, y in zip(val_data['views'], val_data['cart'])]
                # assignment_data["params_1"] = [[x, y, 0.5] for x, y in zip(val_data['views'], val_data['cart'])] #ZeroInflatedPoisson
                # assignment_data["params_1"] = [[x, y, random.uniform(0, 1), random.uniform(0, 1)] for x, y in zip(val_data['views'], val_data['cart'])] #ZeroInflatedNegativeBinomial
                assignment_data.to_hdf("data/assignment_params.h5", key='ap', index=False)
                return assignment_data
                
        params_data = pd.read_hdf("data/assignment_params.h5")
        return params_data


class InitializerTorchDataset(Dataset):
    def __init__(self, value_data, cov_data):
        super().__init__()
        
        self.value_data = value_data
        self.cov_data = cov_data
        
        logging.info("Loading Dataset into Torch Iterator(Initializer)")
        return
    
    def __getitem__(self, index):
        x = torch.tensor(self.cov_data[index, 2:])
        a = torch.tensor(self.value_data[index, [1, 2]])
        y = torch.tensor(self.value_data[index, -1])
        return x, a, y
    
    def __len__(self):
        return self.value_data.shape[0]


class DeconfounderTorchDataset(Dataset):
    def __init__(self, value_data, cov_data, states_data, params_data):
        self.value_data = value_data
        self.cov_data = cov_data
        self.states_data = states_data
        self.params_data = params_data
        
        logging.info("Loading Dataset into Torch Iterator(Deconfounder)")
        return
    
    def __getitem__(self, index):
        z = torch.tensor(self.states_data[index, -1])
        x = torch.tensor(self.cov_data[index, 2:-1])
        ly = torch.tensor(self.cov_data[index, -1])
        a = torch.tensor(self.value_data[index, 2:4])
        y = torch.tensor(self.value_data[index, -1])
        pars = torch.tensor(self.params_data[index, -1])
        
        return a, z, x, y, ly, pars
    
    def __len__(self):
        return self.value_data.shape[0]


class AssignmentTorchDataset(Dataset):
    def __init__(self, value_data):
        self.value_data = value_data
    
    def __getitem__(self, index):
        a1 = torch.tensor(self.value_data[index, 2])
        a2 = torch.tensor(self.value_data[index, 3])
        return a1, a2
    
    def __len__(self):
        return self.value_data.shape[0]


class DeconfounderLightningDataLoader(pl.LightningDataModule):
    def __init__(self, train_params, val_params, train_value, val_value, train_cov, val_cov, train_states, val_states, train_last_value, val_last_value, batch_size=256, test=None):
        super().__init__()
        
        self.user_id = "user_id"
        self.time_col = "event_time_utc"
        self.value_col = ["views", "cart"]
        self.state_col = 'hidden_states'
        
    
        train_value['value'] = np.log(train_value['value'], where=(train_value['value'] != 0))
        val_value['value'] = np.log(val_value['value'], where=(val_value['value'] != 0))
        # train_value['views'] = np.log(train_value['views'], where=(train_value['views'] != 0))
        # val_value['views'] = np.log(val_value['views'], where=(val_value['views'] != 0))
        train_last_value['last_value'] = np.log(train_last_value['last_value'], where=(train_last_value['last_value'] != 0))
        val_last_value['last_value'] = np.log(val_last_value['last_value'], where=(val_last_value['last_value'] != 0))
        
        train_value = pd.merge(train_value, train_cov, on=[self.user_id, self.time_col], how='left')
        train_value = train_value.fillna(0)
        train_value = pd.merge(train_value, train_states.iloc[:, [0, -1]], on=[self.user_id], how='left')
        train_value = train_value.fillna(0)
        train_value = pd.merge(train_value, train_last_value[[self.user_id, "last_value"]], on=[self.user_id], how='left')
        train_value = train_value.fillna(0)
        val_value = pd.merge(val_value, val_cov, on=[self.user_id, self.time_col], how='left')
        val_value = val_value.fillna(0)
        val_value = pd.merge(val_value, val_states.iloc[:, [0, -1]], on=[self.user_id], how='left')
        val_value = val_value.fillna(0)
        val_value = pd.merge(val_value, val_last_value[[self.user_id, "last_value"]], on=[self.user_id], how='left')
        val_value = val_value.fillna(0)
        
        try:
            train_value[self.state_col] = [eval(x) for x in train_value[self.state_col]]
            val_value[self.state_col] = [eval(x) for x in val_value[self.state_col]]
        except Exception:
            pass
        
        try:
            if train_params.params.dtypes == object:
                train_params['params'] = [eval(x) for x in train_params['params']]
            
            if val_params.params.dtypes == object:
                val_params['params'] = [eval(x) for x in val_params['params']]
        except Exception:
            pass
        
        train_value = train_value.dropna()
        val_value = val_value.dropna()
        
        self.train_value = np.array(train_value[[self.user_id, self.time_col] + self.value_col])
        self.train_vectors = np.array(train_value[[x for x in train_value.columns.values if x not in self.value_col and x not in self.state_col]])
        self.train_states = np.array(train_value[[self.user_id, self.time_col, self.state_col]])
        self.val_value = np.array(val_value[[self.user_id, self.time_col] + self.value_col])
        self.val_vectors = np.array(val_value[[x for x in train_value.columns.values if x not in self.value_col and x not in self.state_col]])
        self.val_states = np.array(val_value[[self.user_id, self.time_col, self.state_col]])
        
        self.train_params = np.array(
            pd.merge(train_value[[self.user_id]], train_params, on=self.user_id, how='left')
        )
        self.val_params = np.array(
            pd.merge(val_value[[self.user_id]], val_params, on=self.user_id, how='left')
        )
        
        
        self.setup()

        self.batch_size = batch_size
        return
    
    @property
    def all_users(self):
        return list(set(self.train_value[:, 0]).union(self.val_value[:, 0]))

    def setup(self, stage=None):
        return
        
    def train_dataloader(self):
        return DataLoader(DeconfounderTorchDataset(self.train_value, self.train_vectors, self.train_states, self.train_params), batch_size=self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(DeconfounderTorchDataset(self.val_value, self.val_vectors, self.val_states, self.val_params), batch_size=self.batch_size, num_workers=2)
    
    def all_dataloader(self):
        a = np.append(self.train_value, self.val_value, axis=0)
        print(f"All Data Loaded with {a.shape[0]} rows")
        return DataLoader(DeconfounderTorchDataset(a,
                                                   np.append(self.train_vectors, self.val_vectors, axis=0),
                                                   np.append(self.train_states, self.val_states, axis=0),
                                                   np.append(self.train_params, self.val_params, axis=0)),
                          batch_size=self.batch_size, num_workers=2)


class InitializerLightningDataLoader(pl.LightningDataModule):
    def __init__(self, train_value, val_value, train_cov, val_cov, batch_size=512, test=None):
        super().__init__()
        
        self.user_id = "user_id"
        self.time_col = "event_time_utc"
        self.value_col = "value"
        
        train_value['value'] = np.log(train_value['value'], where=(train_value['value'] != 0))
        val_value['value'] = np.log(val_value['value'], where=(val_value['value'] != 0))
        # train_value['views'] = np.log(train_value['views'], where=(train_value['views'] != 0))
        # val_value['views'] = np.log(val_value['views'], where=(val_value['views'] != 0))
        
        train_value = pd.merge(train_value, train_cov, on=[self.user_id, self.time_col], how='left')
        train_value = train_value.fillna(0)
        val_value = pd.merge(val_value, val_cov, on=[self.user_id, self.time_col], how='left')
        val_value = val_value.fillna(0)
        
        self.train_value = np.array(train_value[[self.user_id, "views", "cart", self.value_col]])
        self.train_vectors = np.array(train_value[[x for x in train_value.columns.values if x != self.value_col]])
        self.val_value = np.array(val_value[[self.user_id, "views", "cart", self.value_col]])
        self.val_vectors = np.array(val_value[[x for x in val_value.columns.values if x != self.value_col]])
        
        self.setup()

        self.batch_size = batch_size
        return

    def setup(self, stage=None):
        return
        
    def train_dataloader(self):
        return DataLoader(InitializerTorchDataset(self.train_value, self.train_vectors), batch_size=self.batch_size, num_workers=10)
    
    def val_dataloader(self):
        return DataLoader(InitializerTorchDataset(self.val_value, self.val_vectors), batch_size=self.batch_size, num_workers=10)
    
    def all_dataloader(self):
        value_data = np.append(self.train_value, self.val_value, axis=0)
        cov_data = np.append(self.train_vectors, self.val_vectors, axis=0)
        avg = np.mean(cov_data[:, 4:], axis=0)
        
        for i in range(4, cov_data.shape[1]):
            cov_data[:, i] = avg[i - 4]

        
        return value_data[:, :2], DataLoader(InitializerTorchDataset(value_data, cov_data), batch_size=self.batch_size, num_workers=10)
