import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn, optim
from torch.autograd import Variable
from get_energy import get_energy
from energy_cost import *

class Pytorch_regression:
    def __init__(self,jobs, epochs=2, doScale=True, n_items=48, verbose=False,plotting=False,return_regret= False,optimizer= optim.SGD,**hyperparam):
        self.n_items = n_items
        self.jobs = jobs

        self.hyperparam = hyperparam
        self.epochs = epochs

        self.doScale=doScale
        self.verbose=verbose
        self.plotting = plotting
        self.return_regret = return_regret
        self.optimizer = optimizer

        self.scaler = None
        self.model = None
        self.best_params_ = {"p":"default"}
        self.criterion = nn.MSELoss()


    def fit(self, x_train, y_train,x_validation=None,y_validation=None,batch=False):
        qids = np.array(x_train[:,0], dtype=int) # qid column
        x_train = x_train[:,1:] # without group ID
        validation = (x_validation is not None) and (y_validation is not None)
        
        # scale data?
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)

        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        
        if self.plotting:
            subepoch_list= []
            loss_list =[]
            regret_list = []
            if validation:
                loss_list_validation= []
                regret_list_validation= []
        # basics
        jobs = self.jobs
        n_items = self.n_items
        n_knapsacks = len(trch_X_train)//n_items
        # prepping
        knaps_V_true = [get_profits(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)]
        knaps_sol = [get_energy_indicators(V_true,jobs) for V_true in knaps_V_true]
        
        # network
        self.model = LinearRegression(trch_X_train.shape[1],1) # input dim, output dim

        # loss
        
        optimizer = self.optimizer(self.model.parameters(), **self.hyperparam)
        num_epochs = self.epochs

        # training
        subepoch = 0 # for logging and nice curves
        logger = [] # (dict_epoch, dict_train, dict_test)
        for epoch in range(num_epochs):
            #scheduler.step()
            knapsack_nrs = [x for x in range(n_knapsacks)]
            random.shuffle(knapsack_nrs) # randomly shuffle order of training
            cnt=0
            if not batch:
                train_fwdbwd(self.model, self.criterion, optimizer, trch_X_train, trch_y_train, 1.0)
            else:
                for kn_nr in knapsack_nrs:
                    V_true = knaps_V_true[kn_nr]
                    kn_start = kn_nr*n_items
                    kn_stop = kn_start+n_items
                    suby=  torch.from_numpy(V_true).float()
                    train_fwdbwd(model= self.model, criterion=self.criterion, 
                             optimizer= optimizer, sub_X_train=trch_X_train[kn_start:kn_stop],
                             sub_y_train= suby, mult=1.0)
            if validation:
                    dict_validation = self.test_score(x_validation,y_validation)
            if self.plotting:
                    cnt += 1
                    subepoch += 1
                    dict_train = test_fwd(self.model, self.criterion, trch_X_train, trch_y_train, n_items, self.jobs)
                    loss_list.append(dict_train['loss'])
                    regret_list.append(dict_train['regret'])
                    subepoch_list.append(subepoch)
                    if validation:
                        loss_list_validation.append(dict_validation['loss'])
                        regret_list_validation.append(dict_validation['regret'])
            if self.verbose:
                        if not self.plotting:
                            cnt += 1
                            subepoch += 1
                        if cnt :
                            dict_epoch = {'epoch': epoch+1, 'subepoch': subepoch, 'cnt': cnt}
                            dict_train = test_fwd(self.model, self.criterion, trch_X_train, trch_y_train, n_items, self.jobs)
                            if validation:
                                logger.append( (dict_epoch, dict_train,dict_validation) )
                                print('Epoch[{}/{}]::{}, loss(train): {:.6f}, regret(train): {:.2f}, loss(validation): {:.6f}, regret(validation): {:.2f}'.format(epoch+1, 
                                    num_epochs, cnt, dict_train['loss'], dict_train['regret'],dict_validation['loss'],dict_validation['regret']  ))
                            else:
                                logger.append( (dict_epoch, dict_train) )
                                print('Epoch[{}/{}]::{}, loss: {:.6f}, regret(train): {:.2f}'.format(epoch+1, num_epochs, cnt, dict_train['loss'], dict_train['regret']))
        
        if self.plotting:
            import matplotlib.pyplot as plt
            if validation:
                plt.subplot(2, 1, 1)
                plt.plot(subepoch_list,regret_list,subepoch_list,regret_list_validation)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top=  np.mean(regret_list)+ 5*np.std(regret_list))
                plt.legend(["training","validation"])
                plt.subplot(2, 1, 2)
                plt.plot(subepoch_list, loss_list,subepoch_list,loss_list_validation)
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.ylim(top= np.median(loss_list)+ 5*np.std(loss_list))
                plt.legend(["training","validation"])
                plt.show()
            else:
                plt.subplot(2, 1, 1)
                plt.plot(subepoch_list,regret_list)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top= np.mean(regret_list)+ 5*np.std(regret_list))
                plt.subplot(2, 1, 2)
                plt.plot(subepoch_list, loss_list)
                plt.ylim(top= np.median(loss_list)+ 5*np.std(loss_list))
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.show()        
        if self.return_regret:
            dict_train = test_fwd(self.model, self.criterion, trch_X_train, trch_y_train, n_items, self.jobs)
            if validation:
                dict_validation = self.test_score(x_validation,y_validation)
                return {"loss_training":dict_train['loss'].item(),"regret_training":dict_train['regret'],
                "loss_validation":dict_validation['loss'],"regret_validation":dict_validation['regret']}
            
            return {"loss":dict_train['loss'].item(),"regret":dict_train['regret']}

    def predict(self, x_test):
        qids = np.array(x_test[:,0], dtype=int) # qid column
        x_test = x_test[:,1:] # drop qid column
        
        # scale data?
        if self.doScale:
            x_test = self.scaler.transform(x_test)

        trch_X = torch.from_numpy(x_test).float()
        pred = self.model(Variable(trch_X))
        return pred.data.numpy().T[0] # as numpy array (transpose)
    def predit_knapsack(self, x_test):
        qids = np.array(x_test[:,0], dtype=int) # qid column
        x_test = x_test[:,1:] # drop qid column

        # scale data?
        if self.doScale:
            x_test = self.scaler.transform(x_test)

        trch_X = torch.from_numpy(x_test).float()
        n_items = self.n_items
        n_knapsacks = len(trch_X)//n_items
        capacity = self.capacity

        knapsack_nrs = [x for x in range(n_knapsacks)]
        pred = []
        for kn_nr in knapsack_nrs:
            V_pred = get_profits_pred(self.model, trch_X, kn_nr, n_items)
            sol_pred = get_kn_indicators(V_pred, capacity, weights=self.weights,use_dp=True)
            pred.append(sol_pred)
        return np.array(pred)

    def test_score(self,x_test,y_test):
        qids = np.array(x_test[:,0],dtype=int)
        x_test = x_test[:,1:] # drop qid column
        # scale data?
        if self.doScale:
            x_test = self.scaler.transform(x_test)
        trch_X = torch.from_numpy(x_test).float()
        trch_y = torch.from_numpy(np.array([y_test]).T).float()
        n_items = self.n_items
        n_knapsacks = len(trch_X)//n_items
 
        dict_test = test_fwd(self.model, self.criterion, trch_X, trch_y, n_items, self.jobs)

        return {"loss":dict_test['loss'].item(),"regret":dict_test['regret'],"confusion_matrix": dict_test['confusion_matrix']}
