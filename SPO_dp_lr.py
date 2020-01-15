import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import torch
from torch import nn, optim
from torch.autograd import Variable
from get_energy import get_energy
from sgd_learner import *
import logging 
import datetime
import time
from collections import defaultdict
class SGD_SPO_dp_lr:
    def __init__(self, capacity=None, weights=None, epochs=2, doScale= True, early_stopping= False,
        n_items=48,model=None, verbose=False,plotting=False,return_regret= False,use_dp= True,use_relaxation=False,validation_relax= False,
        degree=1, optimizer= optim.SGD,store_result =False,**hyperparam):
        self.n_items = n_items
        self.capacity = capacity
        self.weights = weights

        self.hyperparam = hyperparam
        self.epochs = epochs

        self.doScale=doScale
        self.verbose=verbose
        self.plotting = plotting
        self.return_regret = return_regret
        self.optimizer = optimizer
        self.use_dp = use_dp
        self.degree = degree
        self.early_stopping = early_stopping
        self.use_relaxation = use_relaxation
        self.validation_relax = validation_relax
        self.store_result = store_result

        self.scaler = None
        self.model = model
        self.best_params_ = {"p":"default"}
        self.time = 0

    def fit(self, x_train, y_train,x_validation=None,y_validation=None,x_test=None,y_test=None):
        qids = np.array(x_train[:,0], dtype=int) # qid column
        x_train = x_train[:,1:] # without group ID
        validation = (x_validation is not None) and (y_validation is not None)
        test = (x_test is not None) and (y_test is not None)
        if self.early_stopping:
            validation_rslt =[]
        
        # scale data?
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)

        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        if validation:
            x_validation = x_validation[:,1:]
            if self.doScale:
                x_validation = self.scaler.transform(x_validation)
            trch_X_validation = torch.from_numpy(x_validation).float()
            trch_y_validation = torch.from_numpy(np.array([y_validation]).T).float()

        if test:
            x_test = x_test[:,1:]
            if self.doScale:
                x_test = self.scaler.transform(x_test)
            trch_X_test = torch.from_numpy(x_test).float()
            trch_y_test = torch.from_numpy(np.array([y_test]).T).float()

        if self.plotting:
            subepoch_list= []
            loss_list =[]
            regret_list = []
            accuracy_list =[]
            if validation:
                loss_list_validation= []
                regret_list_validation= []
                accuracy_list_validation =[]                    
        
        # basics
        n_items = self.n_items
        n_knapsacks = len(trch_X_train)//n_items
        capacity = self.capacity


        # prepping
        knaps_V_true = [get_profits(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)]
        knaps_sol = [get_kn_indicators(V_true, capacity, weights=self.weights,
            use_dp=self.use_dp,relaxation=self.use_relaxation) for V_true in knaps_V_true]
        for k in knaps_sol:
            self.time+=k[1]

        if validation:
            n_knapsacks_validation = len(trch_X_validation)//n_items
            knaps_V_true_validation = [get_profits(trch_y_validation, kn_nr, n_items) for kn_nr in range(n_knapsacks_validation)]
            knaps_sol_validation = [get_kn_indicators(V_true, capacity, weights=self.weights,use_dp=self.use_dp,relaxation=self.validation_relax) for V_true in knaps_V_true_validation]
            for k in knaps_sol_validation:
                self.time+=k[1]

        if test:
            n_knapsacks_test = len(trch_X_test)//n_items
            knaps_V_true_test = [get_profits(trch_y_test, kn_nr, n_items) for kn_nr in range(n_knapsacks_test)]
            knaps_sol_test = [get_kn_indicators(V_true, capacity, weights=self.weights,use_dp=self.use_dp,relaxation=False) for V_true in knaps_V_true_test]


        # network
        if not self.model:
            self.model = LinearRegression(trch_X_train.shape[1],1) # input dim, output dim

        # loss
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), **self.hyperparam)
        num_epochs = self.epochs

        # training
        subepoch = 0 # for logging and nice curves
        logger = [] # (dict_epoch, dict_train, dict_test)
        test_result =[]
        knapsack_nrs = [x for x in range(n_knapsacks)]
        for epoch in range(num_epochs):
            logging.info('Training Epoch%d Time:%s\n' %(epoch, str(datetime.datetime.now())))
            
            random.shuffle(knapsack_nrs) # randomly shuffle order of training
            cnt = 0
            for kn_nr in knapsack_nrs:
                
                V_true = knaps_V_true[kn_nr]
                sol_true = knaps_sol[kn_nr][0]

                # the true-shifted predictions
                V_pred = get_profits_pred(self.model, trch_X_train, kn_nr, n_items)
                V_spo = (2*V_pred - V_true)

                sol_spo,t = get_kn_indicators(V_spo, capacity,warmstart=sol_true,
                 weights=self.weights,use_dp=self.use_dp,relaxation=self.use_relaxation)
                grad = (sol_spo - sol_true) #*2
                
                if self.degree ==2:
                    sol_pred,t = get_kn_indicators(V_pred,capacity, weights=self.weights,use_dp=self.use_dp,relaxation=self.use_relaxation)
                    reg = sum((sol_true - sol_pred)*V_true)
                    grad = reg*grad
                self.time +=t

                # for each item
                '''for idx in range(len(grad)):
                    pos = idx + (kn_nr*n_items) # indices in train array
                    train_fwdbwd_grad(self.model, optimizer, trch_X_train[pos], trch_y_train[pos], grad[idx])
                '''
                ### what if for the whole 48 items at a time
                kn_start = kn_nr*n_items
                kn_stop = kn_start+n_items
                train_fwdbwd_grad(self.model, optimizer, trch_X_train[kn_start:kn_stop], trch_y_train[kn_start:kn_stop],torch.from_numpy(np.array([grad]).T).float())
                
                if self.verbose or self.plotting or self.store_result:
                    cnt += 1
                    subepoch += 1
                    if cnt % 20 == 0:
                        dict_epoch = {'epoch': epoch+1, 'subepoch': subepoch, 'cnt': cnt}
                        dict_train = test_fwd(self.model, criterion, trch_X_train, trch_y_train, n_items, capacity,knaps_sol,
                            relaxation= self.use_relaxation,weights=self.weights)
                        if validation:
                            dict_validation =  test_fwd(self.model, criterion, trch_X_validation, trch_y_validation, 
                                n_items, capacity,knaps_sol_validation, relaxation = self.validation_relax,  weights=self.weights)
                        if test:
                            dict_test =  test_fwd(self.model, criterion, trch_X_test, trch_y_test, 
                                n_items, capacity,knaps_sol_test, relaxation = False,  weights=self.weights)
                        self.time+= dict_validation['runtime']
                        if self.store_result:
                            info= {}
                            info['train_loss'] = dict_train['loss']
                            info['train_regret_full'] = dict_train['regret_full']
                            info['train_accuracy'] = dict_train['accuracy']
                            info['validation_loss'] = dict_validation['loss']
                            info['validation_regret_full'] = dict_validation['regret_full']
                            info['validation_accuracy'] = dict_validation['accuracy']
                            info['test_loss'] = dict_test['loss']
                            info['test_regret_full'] = dict_test['regret_full']
                            info['test_accuracy'] = dict_test['accuracy']
                            info['subepoch'] = subepoch
                            info['time'] = self.time
                            test_result.append(info)


                        if self.plotting:
                            loss_list.append(dict_train['loss'])
                            regret_list.append(dict_train['regret_full'])
                            accuracy_list.append((dict_train['tn']+dict_train['tp'])/(dict_train['tn']+dict_train['tp']+dict_train['fp']+dict_train['fn']))
                            subepoch_list.append(subepoch)
                            if validation:
                                loss_list_validation.append(dict_validation['loss'])
                                regret_list_validation.append(dict_validation['regret_full'])
                                accuracy_list_validation.append((dict_validation['tn']+dict_validation['tp'])/(dict_validation['tn']+dict_validation['tp']+dict_validation['fp']+dict_validation['fp']))
                        if self.verbose:
                            if validation:
                                logger.append( (dict_epoch, dict_train,dict_validation) )
                                print('Epoch[{}/{}]::{}, loss(train): {:.6f}, regret(train): {:.2f}, loss(validation): {:.6f}, regret(validation): {:.2f}'.format(epoch+1, 
                                    num_epochs, cnt, dict_train['loss'], dict_train['regret_full'],dict_validation['loss'],dict_validation['regret_full']  ))
                            else:
                                logger.append( (dict_epoch, dict_train) )
                                print('Epoch[{}/{}]::{}, loss: {:.6f}, regret(train): {:.2f}'.format(epoch+1, num_epochs, cnt, dict_train['loss'], dict_train['regret_full']))

            if self.early_stopping:
                    dict_train = test_fwd(self.model, criterion, trch_X_train, trch_y_train, n_items, capacity, weights=self.weights)
                    dict_validation =  test_fwd(self.model, criterion, trch_X_validation, trch_y_validation, n_items, capacity, weights=self.weights)
                    validation_rslt.append([epoch, dict_train['loss'].item(), dict_train['regret_full'],dict_validation['loss'].item(),dict_validation['regret_full']])
        if self.plotting:
            import matplotlib.pyplot as plt
            if validation:
                plt.subplot(3, 1, 1)
                plt.plot(subepoch_list,regret_list,subepoch_list,regret_list_validation)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top=  np.mean(regret_list)+ 5*np.std(regret_list))
                plt.legend(["training","validation"])
                plt.subplot(3, 1, 2)
                plt.plot(subepoch_list, loss_list,subepoch_list,loss_list_validation)
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.legend(["training","validation"])
                plt.subplot(3, 1, 3)
                plt.plot(subepoch_list, accuracy_list,subepoch_list, accuracy_list_validation)
                plt.xlabel('Sub Epochs')
                plt.ylabel('Accuracy')
                plt.legend(["training","validation"])
                plt.show()
            else:
                plt.subplot(3, 1, 1)
                plt.plot(subepoch_list,regret_list)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top= np.mean(regret_list)+ 5*np.std(regret_list))
                plt.subplot(3, 1, 2)
                plt.plot(subepoch_list, loss_list)
                plt.yscale('log')
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.subplot(3, 1, 3)
                plt.plot(subepoch_list, accuracy_list)
                plt.ylim(bottom= np.median(accuracy_list)- 3*np.std(accuracy_list))
                plt.xlabel('Sub Epochs')
                plt.ylabel('Accuracy')
                plt.show()           

        if self.early_stopping:
            return pd.DataFrame(validation_rslt,columns=['Epoch','train_loss','train_regret','validation_loss','validation_regret'])
        if self.store_result :
            dd = defaultdict(list)
            for d in test_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            return df

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
            sol_pred = get_kn_indicators(V_pred, capacity, weights=self.weights,use_dp= True)
            pred.append(sol_pred)
        return np.array(pred)

    def test_score(self,x_test,y_test,relaxation=False):
        qids = np.array(x_test[:,0],dtype=int)
        x_test = x_test[:,1:] # drop qid column
        # scale data?
        if self.doScale:
            x_test = self.scaler.transform(x_test)
        trch_X = torch.from_numpy(x_test).float()
        trch_y = torch.from_numpy(np.array([y_test]).T).float()
        n_items = self.n_items
        n_knapsacks = len(trch_X)//n_items
        capacity = self.capacity
        criterion = nn.MSELoss()
        
        dict_test = test_fwd(self.model,criterion,trch_X, trch_y,n_items,capacity,weights = self.weights,relaxation=relaxation)
        return {"loss":dict_test['loss'].item(),"regret":dict_test['regret_full'],"tn": dict_test['tn'],"tp":dict_test['tp'],"fp":dict_test['fp'],"fn":dict_test['fn']}


    # not implemented (should be selective or not?)
    def score(self, attributes, targets):
        return -1

    
    def get_params(self, deep):
        return {'capacity': self.capacity,
                'weights': self.weights,
                'lr': self.lr,
                'mom': self.mom,
                'epochs': self.epochs,
                'doScale': self.doScale}
    
    def set_params(self, capacity=None, weights=None, lr=None, mom=None, epochs=None, doScale=None):
        if capacity != None:
            self.capacity = capacity
        if len(weights) > 0 or weights != None:
            self.weights = weights
        if lr != None:
            self.lr = lr
        if mom != None:
            self.mom = mom
        if epochs != None:
            self.epochs = epochs
        if doScale != None:
            self.doScale = doScale
        return self
class Pytorch_regression:
    def __init__(self, capacity=None, weights=None, epochs=10,early_stopping= False, validation_relax= False,use_dp=True,
        doScale=True, n_items=48,model=None, verbose=False,plotting=False,return_regret= False,store_result =False,
        optimizer= optim.SGD,batch=False,**hyperparam):
        self.n_items = n_items
        self.capacity = capacity
        self.weights = weights
        self.hyperparam = hyperparam
        self.epochs = epochs
        self.early_stopping = early_stopping

        self.doScale=doScale
        self.verbose=verbose
        self.plotting =  plotting
        self.return_regret = return_regret
        self.optimizer = optimizer
        self.batch = batch
        self.validation_relax = validation_relax
        self.store_result = store_result
        self.use_dp = use_dp
        self.time = 0
        self.scaler = None
        self.model = model
        self.batch = batch
        self.criterion = nn.MSELoss()

        self.best_params_ = {"p":"default"}


    def fit(self, x_train, y_train,x_validation=None,y_validation=None,x_test=None,y_test=None):
        batch = self.batch
        qids = np.array(x_train[:,0], dtype=int) # qid column
        x_train = x_train[:,1:] # without group ID
        validation = (x_validation is not None) and (y_validation is not None)
        test = (x_test is not None) and (y_test is not None)
        if self.early_stopping:
            validation_rslt =[]
        
        # scale data?
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)

        trch_X_train = torch.from_numpy(x_train).float()
        trch_y_train = torch.from_numpy(np.array([y_train]).T).float()
        if validation:
            x_validation = x_validation[:,1:]
            if self.doScale:
                x_validation = self.scaler.transform(x_validation)
            trch_X_validation = torch.from_numpy(x_validation).float()
            trch_y_validation = torch.from_numpy(np.array([y_validation]).T).float()

        if test:
            x_test = x_test[:,1:]
            if self.doScale:
                x_test = self.scaler.transform(x_test)
            trch_X_test = torch.from_numpy(x_test).float()
            trch_y_test = torch.from_numpy(np.array([y_test]).T).float()

        if self.plotting:
            subepoch_list= []
            loss_list =[]
            regret_list = []
            accuracy_list =[]
            if validation:
                loss_list_validation= []
                regret_list_validation= []
                accuracy_list_validation =[]                    
        
        # basics
        n_items = self.n_items
        n_knapsacks = len(trch_X_train)//n_items
        capacity = self.capacity
        # prepping
        knaps_V_true = [get_profits(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)]
        knaps_sol = [get_kn_indicators(V_true, capacity, weights=self.weights,
            use_dp=self.use_dp) for V_true in knaps_V_true]


    

        if validation:
            n_knapsacks_validation = len(trch_X_validation)//n_items
            knaps_V_true_validation = [get_profits(trch_y_validation, kn_nr, n_items) for kn_nr in range(n_knapsacks_validation)]
            knaps_sol_validation = [get_kn_indicators(V_true, capacity, weights=self.weights,use_dp=self.use_dp,relaxation=self.validation_relax) for V_true in knaps_V_true_validation]
            for k in knaps_sol_validation:
                self.time+=k[1]

        if test:
            n_knapsacks_test = len(trch_X_test)//n_items
            knaps_V_true_test = [get_profits(trch_y_test, kn_nr, n_items) for kn_nr in range(n_knapsacks_test)]
            knaps_sol_test = [get_kn_indicators(V_true, capacity, weights=self.weights,use_dp=self.use_dp,relaxation=False) for V_true in knaps_V_true_test]


        # network
        if not self.model:
            self.model = LinearRegression(trch_X_train.shape[1],1) # input dim, output dim

        # loss
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), **self.hyperparam)
        num_epochs = self.epochs

        # training
        subepoch = 0 # for logging and nice curves
        logger = [] # (dict_epoch, dict_train, dict_test)
        test_result =[]
        knapsack_nrs = [x for x in range(n_knapsacks)]
        cnt =0

        for epoch in range(num_epochs):
            #scheduler.step()
            
            random.shuffle(knapsack_nrs) # randomly shuffle order of training
           
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

            if self.verbose or self.plotting or self.store_result:
                    cnt += 1
                    subepoch += 1
                    dict_epoch = {'epoch': epoch+1, 'subepoch': subepoch, 'cnt': cnt}
                    dict_train = test_fwd(self.model, criterion, trch_X_train, trch_y_train, n_items, capacity,knaps_sol,
                            weights=self.weights)
                    if validation:
                            dict_validation =  test_fwd(self.model, criterion, trch_X_validation, trch_y_validation, 
                                n_items, capacity,knaps_sol_validation, relaxation = self.validation_relax,  weights=self.weights)
                    if test:
                            dict_test =  test_fwd(self.model, criterion, trch_X_test, trch_y_test, 
                                n_items, capacity,knaps_sol_test, relaxation = False,  weights=self.weights)
                    self.time+= dict_validation['runtime']
                    if self.store_result:
                            info= {}
                            info['train_loss'] = dict_train['loss']
                            info['train_regret_full'] = dict_train['regret_full']
                            info['train_accuracy'] = dict_train['accuracy']
                            info['validation_loss'] = dict_validation['loss']
                            info['validation_regret_full'] = dict_validation['regret_full']
                            info['validation_accuracy'] = dict_validation['accuracy']
                            info['test_loss'] = dict_test['loss']
                            info['test_regret_full'] = dict_test['regret_full']
                            info['test_accuracy'] = dict_test['accuracy']
                            info['subepoch'] = subepoch
                            info['time'] = self.time
                            test_result.append(info)
                    if self.verbose:
                            if validation:
                                logger.append( (dict_epoch, dict_train,dict_validation) )
                                print('Epoch[{}/{}]::{}, loss(train): {:.6f}, regret(train): {:.2f}, loss(validation): {:.6f}, regret(validation): {:.2f}'.format(epoch+1, 
                                    num_epochs, cnt, dict_train['loss'], dict_train['regret_full'],dict_validation['loss'],dict_validation['regret_full']  ))
                            else:
                                logger.append( (dict_epoch, dict_train) )
                                print('Epoch[{}/{}]::{}, loss: {:.6f}, regret(train): {:.2f}'.format(epoch+1, num_epochs, cnt, dict_train['loss'], dict_train['regret_full']))


    
            if self.early_stopping:
                    dict_train = test_fwd(self.model, criterion, trch_X_train, trch_y_train, n_items, capacity, weights=self.weights)
                    dict_validation =  test_fwd(self.model, criterion, trch_X_validation, trch_y_validation, n_items, capacity, weights=self.weights)
                    validation_rslt.append([epoch, dict_train['loss'].item(), dict_train['regret_full'],
                        dict_validation['loss'].item(),dict_validation['regret_full']])

        if self.plotting:
            import matplotlib.pyplot as plt
            if validation:
                plt.subplot(3, 1, 1)
                plt.plot(subepoch_list,regret_list,subepoch_list,regret_list_validation)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top=  np.mean(regret_list)+ 5*np.std(regret_list))
                plt.legend(["training","validation"])
                plt.subplot(3, 1, 2)
                plt.plot(subepoch_list, loss_list,subepoch_list,loss_list_validation)
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.ylim(top=  np.median(loss_list)+3*np.std(loss_list),bottom=0)
                plt.legend(["training","validation"])
                plt.subplot(3, 1, 3)
                plt.plot(subepoch_list, accuracy_list,subepoch_list, accuracy_list_validation)
                plt.legend(["training","validation"])
                plt.xlabel('Sub Epochs')
                plt.ylabel('Accuracy')
                plt.show()            
            else:
                plt.subplot(3, 1, 1)
                plt.plot(subepoch_list,regret_list)
                plt.title('Learning Curve')
                plt.ylabel('Regret')
                plt.ylim(top= np.mean(regret_list)+ 5*np.std(regret_list))
                plt.subplot(3, 1, 2)
                plt.plot(subepoch_list, loss_list)
                plt.ylim(top=  3*np.mean(loss_list))
                plt.xlabel('Sub Epochs')
                plt.ylabel('Loss')
                plt.subplot(3, 1, 3)
                plt.plot(subepoch_list, accuracy_list)
                plt.ylim(bottom= np.median(accuracy_list)- 3*np.std(accuracy_list))
                plt.xlabel('Sub Epochs')
                plt.ylabel('Accuracy')
                plt.show()                            

        if self.return_regret:
            dict_train = test_fwd(self.model, self.criterion, trch_X_train, trch_y_train, n_items, capacity, weights=self.weights)
            if validation:
                dict_validation = self.test_score(x_validation,y_validation)
                return {"loss_training":dict_train['loss'].item(),"regret_training":dict_train['regret_full'],
                "loss_validation":dict_validation['loss'],"regret_validation":dict_validation['regret']}
            
            return {"loss":dict_train['loss'].item(),"regret":dict_train['regret_full']}
        if self.early_stopping:
            return pd.DataFrame(validation_rslt,columns=['Epoch','train_loss','train_regret','validation_loss','validation_regret'])
        if self.store_result :
            dd = defaultdict(list)
            for d in test_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            return df


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
            sol_pred = get_kn_indicators(V_pred, capacity, weights=self.weights)
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
        capacity = self.capacity
        
        dict_test = test_fwd(self.model, self.criterion, trch_X, trch_y, n_items, capacity, weights=self.weights)
        return {"loss":dict_test['loss'].item(),"regret":dict_test['regret_full'],"tn": dict_test['tn'],"tp":dict_test['tp'],"fp":dict_test['fp'],"fn":dict_test['fn']}


    # not implemented (should be selective or not?)
    def score(self, attributes, targets):
        return -1

    
    def get_params(self, deep):
        return {'capacity': self.capacity,
                'weights': self.weights,
                'lr': self.lr,
                'mom': self.mom,
                'epochs': self.epochs,
                'doScale': self.doScale}
    
    def set_params(self, capacity=None, weights=None, lr=None, mom=None, epochs=None, doScale=None):
        if capacity != None:
            self.capacity = capacity
        if len(weights) > 0 or weights != None:
            self.weights = weights
        if lr != None:
            self.lr = lr
        if mom != None:
            self.mom = mom
        if epochs != None:
            self.epochs = epochs
        if doScale != None:
            self.doScale = doScale
        return self
