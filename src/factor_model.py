
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Poisson, NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial, ZeroInflatedPoisson
import pytorch_lightning as pl


class LatentDirichletFactorModel(nn.Module):
    def __init__(self, hidden_states):
        super(LatentDirichletFactorModel, self).__init__()

        self.hidden_states = hidden_states
        
        # Annealing
        T_start = 1.0
        T_end = 0.1
        n_epochs = 15
        temperature_schedule = [T_start - (T_start - T_end) * (i / n_epochs) for i in range(n_epochs)]
        
        self.temperature_schedule = temperature_schedule
        self.model = nn.LSTMCell(
            input_size=self.hidden_states + 1 + 2, hidden_size=self.hidden_states)

    def forward(self, x, t):
        x = self.model(x)[1]
        output = F.softmax(x)
        output = F.softmax(x / self.temperature_schedule[t])
        return output


class LatentFactorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_states):
        super(LatentFactorModel, self).__init__()
        
        self.hidden_states = hidden_states
        params = []
        params.append(nn.LSTMCell(input_size=self.hidden_states + 1, hidden_size=self.hidden_states))
        params.append(nn.Dropout(0.1))
        params.append(nn.Softmax())
        self.model = nn.Sequential(*params)
        

    def forward(self, z):
        alpha = self.model(z)
        return alpha


class AssignmentModel(pl.LightningModule):
    def __init__(self, input_size, with_confounder=True):
        super(AssignmentModel, self).__init__()

        params = []
        params.append(nn.Linear(input_size - 1 + 1, 1))
        params.append(nn.Dropout(0.1))
        params.append(nn.Softplus())
        self.poisson_model = nn.Sequential(*params)  # lambda, a, b
        
        params = []
        params.append(nn.Linear(input_size - 1 + 1, 1))
        params.append(nn.Dropout(0.1))
        params.append(nn.Softplus())
        self.poisson_model_2 = nn.Sequential(*params)  # lambda, a, b
        
        self.poisson_model.apply(self._init_weights)
        self.poisson_model_2.apply(self._init_weights)
    
    def _init_weights(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_normal_(model.weight)
            model.bias.data.fill_(0.01)

    def forward(self, x, pars1, pars2):
        x1 = torch.cat((x, pars1), dim=1)
        x2 = torch.cat((x, pars2), dim=1)
        
        lamb = torch.clamp(torch.nan_to_num(self.poisson_model(x1)), 1e-5, 1000)
        lamb2 = torch.clamp(torch.nan_to_num(self.poisson_model_2(x2)), 1e-5, 1000)
        
        a1 = torch.poisson(lamb)
        a2 = torch.poisson(lamb)

        return torch.cat((lamb, lamb2), dim=1), a1, a2


class SequentialDeconfounder(pl.LightningModule):
    def __init__(self,
                 task: str = "train_model",
                 hidden_states: int = 4,
                 cov_input_dim: int = 15,
                 prior_distribution: dict = {
                     "views": "poisson", "cart": "beta"},
                 prior_hyperparameter: dict = {
                     "poisson": 2, "beta": [0.05, 0.1]},
                 rnn_hidden_size: int = 5,
                 rnn_dropout_rate: float = 0.1,
                 predict_with_deconfounder: bool = True,
                 outcome_dropout_rate: float = 0.1):
        super(SequentialDeconfounder, self).__init__()

        self.task = task
        assert self.task in ['train_model', 'predictive_check', "train_assignment_model", 'output_labels'], f"Unknown parameter for task, expected either 'outcome_model' or 'predictive check', got f{self.task} instead."

        self.prior_distribution = prior_distribution
        self.prior_hyperparameter = prior_hyperparameter

        self.hidden_states = hidden_states
        self.cov_input_dim = cov_input_dim
        self.predict_with_deconfounder = predict_with_deconfounder
        self.assignment_input_dim = self.cov_input_dim + 2 + 2
        self.assignment_input_dim_wo = self.cov_input_dim + 2
        self.rnn_dropout_rate = rnn_dropout_rate
        self.rnn_hidden_size = rnn_hidden_size
        self.outcome_dropout_rate = outcome_dropout_rate

        self._init_models()
        self.loss_func = F.l1_loss

    def _init_weights(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_normal_(model.weight)
            model.bias.data.fill_(0.01)

    def _vec_to_device(self, vec):
        vec = vec.to(device=self.device, dtype=torch.float)
        return vec

    def _init_models(self):
        params = []
        params.append(nn.Linear(self.hidden_states + 1, self.hidden_states))
        params.append(nn.Softmax())
        self.rnn_factor_model = nn.Sequential(*params)
        
        self.LatentFactorModel = LatentDirichletFactorModel(self.hidden_states)
        self.assignment_model = AssignmentModel(
            input_size=self.assignment_input_dim)

        params = []
        params.append(nn.Linear(self.cov_input_dim + 1 + 3, 8))
        # params.append(nn.BatchNorm1d(8))
        params.append(nn.Dropout(self.outcome_dropout_rate))
        params.append(nn.ReLU())
        params.append(nn.Linear(8, 1))
        params.append(nn.ReLU())

        self.outcome_model = nn.Sequential(*params)
        self.outcome_model.apply(self._init_weights)

        params = []
        params.append(nn.Linear(self.cov_input_dim + 3, 4))
        params.append(nn.Dropout(self.outcome_dropout_rate))
        params.append(nn.ReLU())
        params.append(nn.Linear(4, 1))
        params.append(nn.ReLU())

        self.outcome_model_without_deconfounder = nn.Sequential(*params)
        self.outcome_model_without_deconfounder.apply(self._init_weights)
        return

    def _predict_without_deconfounder(self, a, x, ly, y):
        vec = torch.cat((a, x, ly), 1)
        outcome = self.outcome_model_without_deconfounder(vec)
        loss = self.loss_func(outcome, y)
        return outcome, loss

    def _predict_with_deconfounder(self, a, z, x, ly):
        vec = torch.cat((a, z, x, ly), 1)
        outcome = self.outcome_model(vec)
        return outcome

    def forward(self, a, z, x, y, ly, pars):
        a = self._vec_to_device(a)  # assigment variable
        z = self._vec_to_device(z)  # hidden states from last period
        x = self._vec_to_device(x)  # covariates
        y = self._vec_to_device(y).unsqueeze(-1)  # outcome
        ly = self._vec_to_device(ly).unsqueeze(-1)  # outcome last time
        pars1 = self._vec_to_device(pars)[:, 0].unsqueeze(-1)
        pars2 = self._vec_to_device(pars)[:, 1:3]

        # Factor Model: Infer the latent states from RNN cell
        factor = torch.cat((z, ly, pars1, pars2), dim=1)
        state_sample = self.LatentFactorModel(factor, self.current_epoch)  # inferred hidden states
        last_hidden_states = torch.argmax(z, dim=1).unsqueeze(-1)
        hidden_states = torch.argmax(state_sample, dim=1).unsqueeze(-1)
        

        # Assignment Model: Hidden states and covariates will render the Conditional Independence of a1 a2
        vec = torch.cat((x, ly, last_hidden_states, hidden_states), 1)
        param, a1, a2 = self.assignment_model(vec, pars1, pars2)
        
        # Outcome Model
        hidden_states = torch.argmax(state_sample, dim=1).unsqueeze(-1)
        outcome = self._predict_with_deconfounder(a, hidden_states, x, ly)
        
        if self.task == "train_assignment_model":
            return state_sample, a1, a2, param, outcome

        if self.task == "output_labels":
            return state_sample, param

    def _ll_zinb(self, total_count, prob, gate, a):
        zinb = ZeroInflatedNegativeBinomial(total_count=total_count, probs=prob, gate=gate)
        log_prob = zinb.log_prob(a)
        return log_prob.mean()

    def _ll_zpois(self, r, p, a):
        nb = ZeroInflatedPoisson(rate=r, gate=p)
        log_prob = nb.log_prob(a)
        return log_prob.mean()
    
    def _l1_znb(self, r, p, a):
        nb = NegativeBinomial(r, p)
        log_prob = nb.log_prob(a)
        return log_prob.mean()

    def _compute_loss(self, param, a, outcome=None, y=None, with_outcome=False):
        elbo_loss_a1 = - Poisson(param[:, 0]).log_prob(a[:, 0]).mean()
        elbo_loss_a2 = - Poisson(param[:, 1]).log_prob(a[:, 1]).mean()
        if with_outcome:
            outcome_loss = self.loss_func(outcome, y)
            loss = elbo_loss_a1 + elbo_loss_a2 + outcome_loss
            return elbo_loss_a1, elbo_loss_a2, outcome_loss, loss
        else:
            loss = elbo_loss_a1 + elbo_loss_a2
            return elbo_loss_a1, elbo_loss_a2, loss
    
    def training_step(self, batch, batch_idx):
        a, z, x, y, ly, pars = batch
        
        if self.task == "train_model":
            outcome = self.forward(a, z, x, y, ly, pars)
            loss = self.loss_func(outcome, y)
            self.log('training_loss', loss)
        
        if self.task == "train_assignment_model":
            state_sample, a1, a2, param, outcome = self.forward(
                a, z, x, y, ly, pars)
            
            elbo_loss_a1, elbo_loss_a2, outcome_loss, loss = self._compute_loss(param, a, outcome, y, True)
            
            self.log('training_loss', loss)
            self.log('training_outcome_loss', outcome_loss)
            self.log('training_a1_loss', elbo_loss_a1)
            self.log('training_a2_loss', elbo_loss_a2)

        return loss

    def validation_step(self, batch, batch_idx):
        a, z, x, y, ly, pars = batch
        
        if self.task == "train_model":
            outcome = self.forward(a, z, x, y, ly, pars)
            loss = self.loss_func(outcome, y)
            self.log('val_loss', loss)
        
        if self.task == "train_assignment_model":
            state_sample, a1, a2, param, outcome = self.forward(
                a, z, x, y, ly, pars)
            
            elbo_loss_a1, elbo_loss_a2, outcome_loss, loss = self._compute_loss(param, a, outcome, y, True)
            
            self.log('val_loss', loss)
            self.log('val_outcome_loss', outcome_loss)
            self.log('val_a1_loss', elbo_loss_a1)
            self.log('val_a2_loss', elbo_loss_a2)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), weight_decay=0.01, lr=1e-2)
        return optimizer
