import sys
sys.path.insert(0, '..')
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import torch
from sgd_learner import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error as mse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
sys.path.insert(0, '../../EnergyCost/')
# adapted from https://github.com/bwilder0/aaai_melding_code 
class qptl:
    def __init__(self,capacity,weights,tau=20000,doScale= True,n_items=48,epochs=10,
        net=LinearRegression,verbose=False,plotting=False,validation_relax=True,test_relax = False,
        figname=None,validation=False,optimizer=optim.Adam ,**hyperparams):
        self.n_items = n_items
        self.epochs = epochs
        self.net = net
        self.capacity = capacity
        self.weights = weights
        self.model = None
        self.tau = tau
        self.hyperparams = hyperparams
        self.verbose = verbose
        self.plotting = plotting
        self.figname = figname
        self.validation = validation
        self.doScale = doScale
        self.optimizer = optimizer
        self.validation_relax = validation_relax
        self.test_relax = test_relax
        self.model_time = 0.
    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):
        # if validation true validation and tets data should be provided
        tau = self.tau

        start = time.time()
        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided
        X = X[:,1:]
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            X_validation = X_validation[:,1:]
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation -start_validation

        if test:
            start_test = time.time()
            X_test  = X_test[:,1:]
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test


        validation_relax = self.validation_relax
        test_relax = self.test_relax
        n_items = self.n_items
        epochs = self.epochs
        net = self.net
        capacity = self.capacity
        weights = self.weights
        hyperparams = self.hyperparams
        #Q= torch.diagflat(torch.ones(n_items)/tau)
        Q = torch.eye(n_items)/tau
        #G = torch.cat((torch.from_numpy(weights).float(), torch.diagflat(torch.ones(n_items)), 
         # torch.diagflat(torch.ones(n_items)*-1)), 0)
        #h = torch.cat((torch.tensor([capacity],dtype=torch.float),torch.ones(n_items),torch.zeros(n_items)))

        G = torch.from_numpy(weights).float()
        h = torch.tensor([capacity],dtype=torch.float)

        
        self.Q = Q
        self.G= G
        self.h = h        
        
        model = net(X.shape[1],1)
        self.model = model
        #optimizer = torch.optim.Adam(model.parameters(),**hyperparams)
        optimizer = self.optimizer(model.parameters(), **hyperparams)
        model_params_quad = make_gurobi_model(G.detach().numpy(),h.detach().numpy(),None, None, Q.detach().numpy())
        n_knapsacks = X.shape[0]//n_items
        

        loss_list =[]
        accuracy_list =[]
        regret_list = []
        subepoch_list= []
        subepoch= 0
        logger = []
        test_list = []
        n_train = 1 
        for e in range(epochs):
            logging.info('Epoch %d'%e )
            for i in range(n_knapsacks):
                n_start =  n_items*i
                n_stop = n_start + n_items
                z = torch.tensor(y[n_start:n_stop],dtype=torch.float ) 
                X_tensor= torch.tensor(X[n_start:n_stop,:],dtype=torch.float)
                c_true= -z
                c_pred = -(model(X_tensor))
                solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)
                x = solver(Q.expand(n_train, *Q.shape),
                c_pred.squeeze(), G.expand(n_train, *G.shape), 
                h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())

                self.model_time +=solver.Runtime()
                loss = (x.squeeze()*c_true).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                subepoch += 1
                if i%20==0:
                    if self.verbose:
                        dict_validation = {}
                        logging.info('Validation starts\n ' )
                        #train_result = self.test(X,y)
                        
                        ##### to be deleted ###########
                        # train_result = self.test(X  ,y,relaxation = False)
                        # dict_validation['training_regret'] = train_result[0]
                        # dict_validation['training_mse'] = train_result[1]
                        # dict_validation['training_accuracy'] = train_result[2]
                        # dict_validation['training_loss'] = train_result[3] 
                        # print("Loss: at Epoch %d is %f Regret %f "%(subepoch,train_result[3],train_result[0]))
                        ############################ ###################                       
                        if validation:
                            start_validation = time.time()
                            validation_result = self.test(X_validation,y_validation,relaxation = validation_relax)
                            self.model_time+= validation_result[4]
                            dict_validation['validation_regret'] = validation_result[0]
                            dict_validation['validation_mse'] = validation_result[1]
                            dict_validation['validation_accuracy'] = validation_result[2]
                            dict_validation['validation_loss'] = validation_result[3]
                            end_validation = time.time()
                            validation_time += end_validation -start_validation
                        if test:
                            start_test = time.time()
                            test_result = self.test(X_test,y_test , relaxation = test_relax)
                            self.model_time+= validation_result[4]
                            dict_validation['test_regret'] = test_result[0]
                            dict_validation['test_mse'] = test_result[1]
                            dict_validation['test_accuracy'] = test_result[2]
                            dict_validation['test_loss'] = test_result[3]
                            end_test = time.time()
                            test_time+=  end_test - start_test


                        dict_validation['subepoch'] = subepoch
                        dict_validation['Runtime'] = self.model_time
                        dict_validation['time'] = time.time() - start  



                        
                        test_list.append(dict_validation)

                        logging.info("Epoch %d::subepoch %d Total time %d, validation time %d & test time %d"%(e,
                            i,time.time() - start,validation_time,test_time))
                    
                        #print('Epoch[{}/{}], loss(train):{:.2f} '.format(e+1, i, loss.item() ))
                    if self.plotting:
                        
                        subepoch_list.append(subepoch)
                        reg,loss,acc = self.test(X,y)
                        loss_list.append(loss)
                        regret_list.append(reg)
                        accuracy_list.append(acc)
        if self.plotting:
            fig, (ax1, ax2,ax3) = plt.subplots(3,1,figsize=(6, 6))
            ax1.plot(subepoch_list,regret_list)
            ax1.set_ylabel('Regret')
            ax2.plot(subepoch_list,loss_list)
            ax2.set_yscale('log')
            ax2.set_ylabel('Loss')
            ax3.plot(subepoch_list,accuracy_list)
            ax3.set_xlabel('Sub Epochs')
            ax3.set_ylabel('Accuracy')
            plt.savefig(self.figname)
        if self.verbose:
            dd = defaultdict(list)
            for d in test_list:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            return df

    def pred(self,X):
        X = X[:,1:]
        if self.doScale:
            X = self.scaler.transform(X)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X,dtype=torch.float)
        model.train()
        return model(X_tensor).detach().numpy().squeeze()
    def test(self,X,y,relaxation=False):
        Q= self.Q
        G = self.G
        h =  self.h
        n_train = 1 
        n_items = self.n_items
        epochs = self.epochs
        net = self.net
        capacity = self.capacity
        model = self.model
        weights = self.weights
        model.eval()
        X_tensor= torch.tensor(X,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        n_knapsacks = X.shape[0]//n_items
        regret_list= []
        cf_list = []
        loss_list = []
        time = 0
        model_params_quad = make_gurobi_model(G.detach().numpy(),h.detach().numpy(),None, None, Q.detach().numpy())
        for i in range(n_knapsacks):
            n_start =  n_items*i
            n_stop = n_start + n_items
            regret, cf=regret_knapsack([y[n_start:n_stop]],[y_pred[n_start:n_stop]],
                weights=weights,cap=[self.capacity],relaxation = relaxation)
            regret_list.append(regret)
            cf_list.append(cf)
            z = torch.tensor(y[n_start:n_stop],dtype=torch.float )
            X_tensor= torch.tensor(X[n_start:n_stop,:],dtype=torch.float) 
            c_true= -z
            c_pred = -(model(X_tensor))
            solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)
            x = solver(Q.expand(n_train, *Q.shape),
                c_pred.squeeze(), G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
            time +=solver.Runtime()        
            loss_list.append((x.squeeze()*c_true).mean().item())
        model.train()
        if not relaxation:
            tn, fp, fn, tp = np.sum(np.stack(cf_list),axis=0).ravel()
            accuracy = (tn+tp)/(tn+fp+fn+tp)
        else:
            accuracy = None        

        return np.median(regret_list), mse(y,y_pred), accuracy,np.median(loss_list),time





