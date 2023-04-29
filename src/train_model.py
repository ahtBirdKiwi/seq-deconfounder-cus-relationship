import os
import shutil

import numpy as np
import pandas as pd
import scipy.stats as sts
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_loader import InitializerLightningDataLoader, DeconfounderLightningDataLoader, IterativeLoader
from src.state_initiailizer import LinearEstimator, init_by_ordering
from src.factor_model import SequentialDeconfounder


def full_run():
    IL = IterativeLoader()
    
    last_value = IL.get_data(1)[0]
    last_value = last_value[["user_id", "event_time_utc", "value"]]
    last_value.columns = ["user_id", "event_time_utc", "last_value"]
    
    BSD = BuildSequentialDeconfounder()
    p_val_negs = []
    last_assignments = None
    for cnt in range(2, 7):
        if cnt > 1:
            all_states = IL.get_hidden_states()
            all_params = IL.get_assignment_params()
            all_params = all_params[['user_id', f'params_{cnt - 1}']]
            all_params.columns = ["user_id", "params"]
            
            if cnt > 2:
                BSD.predictive_checks(all_params, last_assignments)  # Predictive checks using last_assignments
                p_val_neg = BSD.p_vals[-1]
                p_val_negs.append((cnt, p_val_neg))
                print(f"Time = {cnt-1}, Pred_Checks = {p_val_neg}")
            
            dataset = IL.get_data(cnt)
            value, covariate = dataset[0], dataset[1]
            del dataset
            
            states = all_states[['user_id', f'hidden_states_{cnt - 1}']]
            states.columns = ["user_id", "hidden_states"]
            del all_states
            print(f"Time = {cnt}")
            BSD.load_data(cnt, all_params, value, covariate, states, last_value)
            del covariate, states
            
            BSD.fit_assignment()
            BSD.output(timer=cnt)
            
            if cnt == 6:
                all_params = IL.get_assignment_params()
                all_params = all_params[['user_id', f'params_{cnt - 1}']]
                all_params.columns = ["user_id", "params"]
                BSD.predictive_checks(all_params, last_assignments)  # Predictive checks using last_assignments
                p_val_neg = BSD.p_vals[-1]
                p_val_negs.append((cnt, p_val_neg))
                print(f"Time = {cnt}, Pred_Checks = {p_val_neg}")
            
            
            last_value = value.copy()
            last_assignments = BSD.val_value
            last_value = last_value[["user_id", "event_time_utc", "value"]]
            last_value.columns = ["user_id", "event_time_utc", "last_value"]
    
    print(p_val_negs)


class PredictiveChecks():
    def __init__(self, param_data, real_data, n_eval=1000, cut=100000):
        # real_data: true outcome
        # param_data: the estimated parameters
        self.n_eval = n_eval
        
        self.all_data = pd.merge(real_data[['user_id', 'views', 'cart']], param_data.iloc[:, [0, -1]], how='left', on='user_id')
        self.all_data = self.all_data.sample(cut)
        
        # handle data format issue
        try:
            self.all_data['params'] = [eval(x) for x in self.all_data['params']]
        except Exception:
            pass
        
    def run(self):
        # Get parameters
        lambda1 = [x[0] for x in self.all_data['params']]
        lambda2 = [x[1] for x in self.all_data['params']]
        
        # Calculate holdout log probabilitiesz
        a1_holdout = sts.poisson(lambda1).logpmf(self.all_data['views'])
        a2_holdout = sts.poisson(lambda2).logpmf(self.all_data['views'])

        # Generate replicated data and calculate log probabilities
        poisson_samples = np.random.poisson(lambda1, size=(self.n_eval, len(lambda1)))
        a1_stats = sts.poisson(lambda1).logpmf(poisson_samples)
        poisson_samples_2 = np.random.poisson(lambda2, size=(self.n_eval, len(lambda1)))
        a2_stats = sts.poisson(lambda2).logpmf(poisson_samples_2)
 
        # Calculate p-values
        rep_a1 = np.mean(np.mean(np.where(a1_stats <= a1_holdout, 1, 0), axis=1))
        rep_a2 = np.mean(np.mean(np.where(a2_stats <= a2_holdout, 1, 0), axis=1))
        
        return rep_a1, rep_a2


class BuildSequentialDeconfounder():
    def __init__(self, hidden_states=3, max_epochs=4, min_delta=1e-2, patience=1, inited=False, with_deconfounder=True):
        self.save_path = "model_params/"
        
        self.hidden_states = hidden_states
        assert self.hidden_states > 1, "parameter hidden_states should be at least greater than 1"
        
        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.patience = patience
        self.with_deconfounder = with_deconfounder
        
        self.inited = inited
    
    def load_data(self, time, params, value, covariate, states, last_value):
        print("Loading Data...")
        self.time = time
        self._train_val_split(params, value, covariate, states, last_value)
        print("Loading Data Done!")

    def _create_dir(self):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
    
    def _clean_lightning_log(self):
        if os.path.isdir('model_params/deconfounder/assignment/lightning_logs/'):
            shutil.rmtree('model_params/deconfounder/assignment/lightning_logs/')
        
        if not os.path.isdir('model_params/deconfounder/assignment/lightning_logs/'):
            os.mkdir('model_params/deconfounder/assignment/lightning_logs/')
            
        return
    
    def _train_val_split(self, params, value, covariate, states, last_value, split_ratio=0.8):
        all_users_last_time = last_value['user_id']
        value = value[value['user_id'].isin(all_users_last_time)]
        
        train_value, val_value = train_test_split(value, test_size=1 - split_ratio)
        train_cov = covariate[covariate['user_id'].isin(train_value["user_id"])]
        val_cov = covariate[covariate['user_id'].isin(val_value["user_id"])]
        train_states = states[states['user_id'].isin(train_value["user_id"])]
        val_states = states[states['user_id'].isin(val_value["user_id"])]
        train_last_value = last_value[last_value['user_id'].isin(train_value['user_id'])]
        val_last_value = last_value[last_value['user_id'].isin(val_value['user_id'])]
        train_params = params[params['user_id'].isin(train_value["user_id"])]
        val_params = params[params['user_id'].isin(val_value["user_id"])]
        
        self.dataset = DeconfounderLightningDataLoader(train_params, val_params, train_value, val_value, train_cov,
                                                       val_cov, train_states, val_states, train_last_value, val_last_value)
        self.val_value = val_value.copy()
        print(f"Loading Data Done with {len(self.dataset.all_users)} users")
        return

    def _init_dataset_and_model(self, type_trainer="outcome"):
        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=False,
            mode='min'
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_path,
            save_top_k=0,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        
        self.ckpt_path = self.save_path + 'deconfounder/' if self.with_deconfounder else self.save_path + 'no_deconfounder/'
        resume = self.ckpt_path + '_assignment.ckpt' if type_trainer == "assignment" else self.ckpt_path + "_outcome.ckpt"
       
        if not self.inited:
            print("Initiate Model")
            self.v_num = 0
            self.tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.ckpt_path + "/" + type_trainer + "/")
            
            self.model = SequentialDeconfounder(hidden_states=self.hidden_states, predict_with_deconfounder=self.with_deconfounder)
            self.trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[self.early_stop_callback, self.checkpoint_callback], num_sanity_val_steps=0, logger=self.tb_logger, detect_anomaly=True)
            self.loop_epochs = self.max_epochs
            
            if os.path.exists(resume):
                os.remove(resume)
            if os.path.exists(resume.replace(".ckpt", ".pkl")):
                os.remove(resume.replace(".ckpt", ".pkl"))
            
            self.p_vals = []
            self.inited = True
        else:
            self.loop_epochs += self.max_epochs
            self.v_num += 1
            self.trainer = pl.Trainer(max_epochs=self.loop_epochs, callbacks=[self.early_stop_callback, self.checkpoint_callback], num_sanity_val_steps=0, resume_from_checkpoint=resume, logger=self.tb_logger, detect_anomaly=True)

    def fit_assignment(self, save=True):
        self._init_dataset_and_model(type_trainer="assignment")
        self.model.task = "train_assignment_model"
        self.model.train()
        
        self.trainer.fit(self.model, self.dataset)
        self.trainer.save_checkpoint(self.ckpt_path + '_assignment.ckpt')
        return
    
    def save(self):
        self.trainer.save_checkpoint(self.ckpt_path + '_assignment.ckpt')
        torch.save(self.model, self.ckpt_path + '_assignment.pkl')
        return
    
    def fit(self, save=True):
        self._init_dataset_and_model(type_trainer="outcome")

        self.model.task = "train_model"
        self.model.train()
        self.trainer.fit(self.model, self.dataset)
        
        if save:
            self.trainer.save_checkpoint(self.ckpt_path + '_outcome.ckpt')
            torch.save(self.model, self.ckpt_path + '_outcome.pkl')
        return
    
    def predictive_checks(self, new_params, last_assignment):
        self._init_dataset_and_model()
        
        PC = PredictiveChecks(new_params, last_assignment)
        a1_p, a2_p = PC.run()
        self.p_vals.append([a1_p, a2_p])
        return
    
    def output(self, timer: int):
        if not self.inited:
            self._init_dataset_and_model()
        
        self.model.task = "output_labels"
        self.model.eval()
        
        AllLoader = self.dataset.all_dataloader()
        AllUsers = pd.DataFrame(self.dataset.all_users, columns=['user_id'])
        print(f"Number of Users = { AllUsers.shape[0] }")
        
        States = []
        Params = []
        for idx, batch in enumerate(tqdm(AllLoader, desc='Get Labels...')):
            a, z, x, y, ly, pars = batch
            state_sample, param = self.model.forward(a, z, x, y, ly, pars)

            States.extend(list(state_sample.cpu().detach().numpy()))
            Params.extend(list(param.cpu().detach().numpy()))

        AllUsers['hidden_states_' + str(timer)] = States
        AllUsers['hidden_states_' + str(timer)] = [list(x) for x in AllUsers['hidden_states_' + str(timer)]]
        
        AllUsers['params_' + str(timer)] = Params
        AllUsers['params_' + str(timer)] = [list(x) for x in AllUsers['params_' + str(timer)]]
        
        del AllLoader, States, Params
        
        hidden_states = pd.read_hdf("data/hidden_states.h5")
        
        if "hidden_states_" + str(timer) not in hidden_states.columns.values:
            hidden_states = pd.merge(hidden_states, AllUsers[['user_id', 'hidden_states_' + str(timer)]], on=['user_id'], how='left')
            hidden_states.to_hdf("data/hidden_states.h5", key='hs', index=False)

        del hidden_states
        pars = pd.read_hdf("data/assignment_params.h5")
        
        if "params_" + str(timer) not in pars.columns.values:
            pars = pd.merge(pars, AllUsers[['user_id', 'params_' + str(timer)]], on=['user_id'], how='left')
            pars.to_hdf("data/assignment_params.h5", key='ap', index=False)
        
        return
    
    
class InitHiddenStates():
    def __init__(self, value, covariate, hidden_states=3, max_epochs=2, min_delta=1e-3, patience=1):
        self.save_path = "model_params/"
        self.hidden_states = hidden_states
        
        assert self.hidden_states > 1, "parameter hidden_states should be at least greater than 1"
        
        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.patience = patience
        self.input_dim = value.shape[1] + covariate.shape[1] - 5  # -4: (user_id, event_time_utc) * 2
        
        self._clean_lightning_log()
        self._train_val_split(value, covariate)
        self.inited = False
        return
    
    
    def do_naive_ordering(self):
        init_by_ordering(pd.concat([self.train_value, self.val_value]))
        return
    
    def _create_dir(self):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
    
    def _clean_lightning_log(self):
        if os.path.isdir('lightning_logs/'):
            shutil.rmtree('lightning_logs/')
        
        if not os.path.isdir('lightning_logs/'):
            os.mkdir('lightning_logs/')
            
        return
    
    def _train_val_split(self, value, covariate, split_ratio=0.8):
        self.train_value, self.val_value = train_test_split(value, test_size=1 - split_ratio)
        self.train_cov = covariate[covariate['user_id'].isin(self.train_value["user_id"])]
        self.val_cov = covariate[covariate['user_id'].isin(self.val_value["user_id"])]
        
        return

    def _init_dataset_and_model(self):
        self.dataset = InitializerLightningDataLoader(self.train_value, self.val_value, self.train_cov, self.val_cov)
        self.model = LinearEstimator(hidden_states=self.hidden_states, input_dim=self.input_dim)
        
        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=False,
            mode='min'
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_path,
            save_top_k=0,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        
        self.inited = True
    
    def fit(self, save=True):
        self._init_dataset_and_model()
        
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[self.early_stop_callback, self.checkpoint_callback], gradient_clip_val=0.5)
        self.trainer.fit(self.model, self.dataset)
        
        if save:
            self.trainer.save_checkpoint(self.save_path + 'linear_estimator.ckpt')
            torch.save(self.model, self.save_path + 'linear_estimator.pkl')

    def validate(self):
        if not self.inited:
            self._init_dataset_and_model()
        
        self.model = torch.load(self.save_path + 'linear_estimator.pkl')
        self.model.eval()
        torch.no_grad()
        
        rt = pd.DataFrame(columns=['true', 'pred'])
        dataset = self.dataset.val_dataloader()
        
        for idx, batch in enumerate(tqdm(dataset, desc='Validate Model...')):
            vec, a, y = batch
            a = self.model._vec_to_device(a)
            vec = self.model._vec_to_device(vec)
            
            output = self.model(vec, a, y)[0]
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            
            rrt = pd.DataFrame(columns=['true', 'pred'])
            rrt['true'] = y
            rrt['pred'] = output
            rt = pd.concat([rt, rrt])


    def label_states(self):
        if not self.inited:
            self._init_dataset_and_model()
        
        self.model = torch.load(self.save_path + 'linear_estimator.pkl')
        self.model.eval()
        torch.no_grad()
        user_time, dataset = self.dataset.all_dataloader()
        
        preds = []

        for idx, batch in enumerate(tqdm(dataset, desc='Get Labels...')):
            vec, a, y = batch
            vec = self.model._vec_to_device(vec)
            a = self.model._vec_to_device(a)
            
            output = self.model(vec, a, y)[0]
            output = output.cpu().detach().numpy()
            preds.extend(list(output))
        
        user_time = np.append(user_time, preds, axis=1)
        user_time = pd.DataFrame(user_time, columns=['user_id', 'event_time_utc', 'states'])

        user_time_0 = user_time[user_time['states'] == 0]
        user_time_0['hidden_state'] = 0
        user_time_0 = user_time_0[['user_id', 'event_time_utc', 'hidden_state']]
        
        user_time_non_0 = user_time[user_time['states'] > 0]
        user_time_non_0['hidden_state'] = pd.cut(user_time_non_0['states'], self.hidden_states - 1, labels=[i + 1 for i in range(self.hidden_states - 1)])
        user_time_non_0 = user_time_non_0[['user_id', 'event_time_utc', 'hidden_state']]
        
        user_time = pd.concat([user_time_0, user_time_non_0])
        del user_time_0, user_time_non_0
        
        user_time['hidden_states'] = [[0] * self.hidden_states for _ in range(user_time.shape[0])]
        for sta in range(user_time.shape[0]):
            user_time.iloc[sta, -1][user_time.iloc[sta, -2]] = 1
        
        user_time = user_time[['user_id', 'hidden_states']]
        user_time.columns = ['user_id', 'hidden_states_1']

        user_time.to_hdf('data/hidden_states.h5', key='hs', index=False)
        return
