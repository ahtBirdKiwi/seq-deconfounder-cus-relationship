import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


def weightedMSE(output, target):
    weights = torch.tensor(torch.where(target == 0, 0.3, 1))
    return torch.mean(weights * (output - target) ** 2)


def init_by_ordering(value):
    users = pd.DataFrame(value['user_id'], columns=['user_id'])

    # apply qcut to entries with value > 0 and update the label column
    mask = value['value'] > 0
    users.loc[mask, 'hidden_states_1'] = pd.qcut(value.loc[mask, 'value'], q=2, labels=[1, 2])
    users['hidden_states_1'] = users['hidden_states_1'].cat.add_categories([0])
    users.fillna(0, inplace=True)
    # print(users)
    users['hidden_states_1'] = users['hidden_states_1'].astype(int)
    at = [[0] * 3 for _ in range(users.shape[0])]
    for i in range(len(at)):
        at[i][users.iloc[i, -1]] = 1
    users['hidden_states_1'] = [str(x) for x in at]
    users.to_hdf("data/hidden_states.h5", key='hs', index=False)


class LinearEstimator(pl.LightningModule):
    def __init__(self, hidden_states: int, input_dim: int, zero_penalty=0.2):
        super(LinearEstimator, self).__init__()
        
        self.hidden_states = hidden_states
        self.input_dim = input_dim
        
        params = []
        params.append(nn.Linear(self.input_dim + 2, 1))
        # params.append(nn.BatchNorm1d(4))
        params.append(nn.Dropout(0.1))
        params.append(nn.ReLU())
        # params.append(nn.Linear(4, 1))
        # params.append(nn.ReLU())
        
        self.FC = nn.Sequential(*params)
        # self.loss_func = weightedMSE
        self.loss_func = nn.L1Loss()
        
    
    def _init_weights(self, model):
        if type(model) == nn.Linear:
            nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    def _vec_to_device(self, vec):
        vec = vec.to(device=self.device, dtype=torch.float)
        return vec
    
    def forward(self, vec, a, y=None):
        vec = self._vec_to_device(vec)
        a = self._vec_to_device(a)
        vec = torch.cat((vec, a), dim=1)
        
        if y is not None:
            y = self._vec_to_device(y).unsqueeze(-1)
        
        output = self.FC(vec)
        output = torch.where(torch.isnan(output), torch.full_like(output, 1e-6), output)
        
        if y is not None:
            loss = self.loss_func(output, y)
            return output, loss
        else:
            return
    
    def training_step(self, batch, batch_idx):
        vec, a, y = batch

        total, loss = self.forward(vec, a, y)
        self.log('training_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        vec, a, y = batch

        total, loss = self.forward(vec, a, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), weight_decay=0.1, lr=1e-3)
        return optimizer
