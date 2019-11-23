from __future__ import print_function
import time
from ortools.algorithms import pywrapknapsack_solver
import numpy as np
from scipy import stats
from sklearn.metrics.scorer import _BaseScorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import collections

# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn, optim
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)  # input and output is 1 dimension
        

    def forward(self, x):
        out = self.linear(x)
        return out

class LogitRegression(nn.Module):
    def __init__(self, dim_in, num_classes):
        super().__init__()
        self.linear = nn.Linear(dim_in, num_classes)  # input and output is 1 dimension
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        out1 = self.linear(x)
        out2 = self.softmax(out1)
        return out2
    
    def take_outY(self,x):
        self.train(False)
        return self.linear(x)


def cost_minimize(price,jobs):
    # price is a sequnce of prices
    # jobs is a list of n_job where each item describes processing time of the job
    jobs_count = len(jobs)
    all_jobs = range(jobs_count)
    all_tasks = {}
    horizon = range(len(price))
    model = cp_model.CpModel()
    machines= {}
    machine_start = {}
    multiplier = 1e+2
    price = [int(v*multiplier) for v in price]
    for job in all_jobs:
        for  d in horizon:
            machines[job,d] = model.NewIntVar(0, 1, 'On_%i_%i' % (job, d))
            machine_start[job,d] = model.NewIntVar(0, 1, 'start_%i_%i' % (job, d))
            if d==0:
                model.Add(machine_start[job,d]==machines[job,d])
            else:
                model.Add(machine_start[job,d]>= (machines[job,d]-machines[job,d-1]))
    for job in all_jobs:
                model.Add(sum(machines[job,d] for d in (horizon)) == jobs[job])
                model.Add(sum(machine_start[job,d] for d in (horizon)) == 1)
    model.Minimize(sum([sum([machines[job,d]*price[d] for d in (horizon)]) for job in all_jobs]))
    solver = cp_model.CpSolver()
    solver.Solve(model)
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        objective = solver.ObjectiveValue()/multiplier
        assignment= list()
        for job in all_jobs:
            allocation= list()
            for d in horizon:
                allocation.append( solver.Value(machines[job,d]))
            assignment.append(allocation)
        return {'objective':objective, 'assignments':assignment}
    else:
        print("ERROR HAPPENED!\n")
        print(price)
        return {'objective':math.inf, 'assignments':[[1 for d in horizon] for jon in all_jobs] }
def get_energy_indicators(price,jobs):
    return np.array( cost_minimize(price,jobs)['assignments'][0])
def get_profits(trch_y, kn_nr, n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items
    return trch_y[kn_start:kn_stop].data.numpy().T[0]

def get_profits_pred(model, trch_X, kn_nr, n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items
    model.eval()
    with torch.no_grad():
        V_pred = model(Variable(trch_X[kn_start:kn_stop]))
    model.train()
    return V_pred.data.numpy().T[0]
    
def train_fwdbwd_grad(model, optimizer, sub_X_train, sub_y_train, grad):
    inputs = Variable(sub_X_train, requires_grad=True)
    target = Variable(sub_y_train)
    out = model(inputs)
    grad = grad*torch.ones(1)
    
    optimizer.zero_grad()
    
    # backward
    # hardcode the gradient, let the automatic chain rule backwarding do the rest
    loss = out
    loss.backward(gradient=grad)
    
    optimizer.step()
def train_fwdbwd(model, criterion, optimizer, sub_X_train, sub_y_train, mult):
    inputs = Variable(sub_X_train)
    target = Variable(sub_y_train)
    out = model(inputs)
    # weighted loss...
    loss = torch.tensor(mult)*criterion(out, target)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
def train_fwdbwd_oneitem(model, criterion, optimizer, trch_X_train, trch_y_train, pos, mult):
    train_fwdbwd(model, criterion, optimizer, trch_X_train[pos], trch_y_train[pos], mult)

    
def test_fwd(model, criterion, trch_X, trch_y, n_items,jobs):
    info = dict()
    from sklearn.metrics import confusion_matrix
    model.eval()
    with torch.no_grad():
        # compute loss on whole dataset
        inputs = Variable(trch_X)
        target = Variable(trch_y)
        V_preds = model(inputs)
        info['loss'] = criterion(V_preds, target).data
    model.train()
        
    n_knap = len(V_preds)//n_items
    regret= np.zeros(n_knap)
    cf_list =[]
    # I should probably just slice the trch_y and preds arrays and feed it like that...
    for kn_nr in range(n_knap):
        V_true = get_profits(trch_y, kn_nr, n_items)
        V_pred = get_profits(V_preds, kn_nr, n_items)
            
        sol_true = get_energy_indicators(V_true,jobs)
        sol_pred = get_energy_indicators(V_pred,jobs)
        regret[kn_nr] = sum(V_true*(sol_pred - sol_true))
        cf = confusion_matrix(sol_true, sol_pred,labels=[0,1])
        cf_list .append(cf)

    info['regret'] = np.average(regret)
    info['confusion_matrix'] = np.sum(np.stack(cf_list),axis=0)
    return info
def diffprof(V_pred, index, newvalue, V_true,jobs):
    sol = get_energy_indicators(V_pred,jobs)
    
    Vnew = np.array(V_pred)
    Vnew[index] = newvalue
    sol_new = get_energy_indicators(Vnew,jobs)
    return sum(V_true*(sol_new - sol)) # difference in obj
