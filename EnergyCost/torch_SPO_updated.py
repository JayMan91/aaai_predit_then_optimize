import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn, optim
from torch.autograd import Variable
import logging
import sys
sys.path.insert(0,'..')
import datetime
import time

from sgd_learner import *
from sklearn.metrics import confusion_matrix
from collections import defaultdict
class SGD_SPO_generic:
    def __init__(self,param,solver,reset,presolve,relax=False, validation_relax=False,test_relax=False, optimal_value=knapsack_value,getY= get_profits,getTorchData= get_data, 
        getYpred = get_profits_pred, epochs=2, doScale= True, n_items=48,model=None,timelimit = None,
        verbose=False,plotting=False,degree=1, maximize= True,accuracy_measure=  False,early_stopping= False,
        optimizer= optim.SGD,model_save=False,model_name=None,method=-1,warmstart=False,# obj_cut=-1,
       **hyperparam):

        #-1 : no objective cut
        # 0: cut for predictions only 'true' solution
        # n: previous n solutions as cut
        self.n_items = n_items        
        self.hyperparam = hyperparam
        self.epochs = epochs

        self.doScale=doScale
        self.verbose=verbose
        self.plotting = plotting
        self.optimizer = optimizer
        self.degree = degree
        #self.solver_train = solver_train
        #self.solver_test = solver_test
        self.solver =  solver
        self.relax = relax
        self.validation_relax = validation_relax
        self.test_relax = test_relax
        self.optimal_value = optimal_value
        self.param = param
        self.maximize = maximize
        self.getY =  getY
        self.getYpred = getYpred
        self.accuracy_measure = accuracy_measure
        self.getTorchData = getTorchData
        self.early_stopping = early_stopping
        self.true_solution = None
        self.validation_solution = None
        self.timelimit = timelimit
        self.time = 0
        self.scaler = None
        self.criterion = None
        self.model = model
        self.model_save = model_save
        self.model_name = model_name
        self.method = method
        self.reset = reset
        self.presolve = presolve
        self.warmstart = False
        
        #self.obj_cut = obj_cut
        #self.sol_hist = []
        self.best_params_ = {"p":"default"}

        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('info.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Completed configuring logger()!')
        '''


    def test_fwd(self, criterion, trch_X, trch_y,accuracy):
        from sklearn.metrics import confusion_matrix
        info = dict()
        n_items = self.n_items
        solver= self.solver
        optimal_value = self.optimal_value
        param= self.param
        clf_train = self.clf_train
        clf_validation = self.clf_validation
        clf_test = self.clf_test
        max_min_ind = 1 if self.maximize else -1

        self.model.eval()
        with torch.no_grad():
            inputs = Variable(trch_X)
            target = Variable(trch_y)
            V_preds = self.model(inputs)
            info['loss'] = criterion(V_preds, target).data
        self.model.train()
        n_knap = len(V_preds)//n_items
        regret_full = np.zeros(n_knap)
        cf_list =[]

        for kn_nr in range(n_knap):
            V_true = self.getY(trch_y, kn_nr, n_items)
            V_pred = self.getY(V_preds, kn_nr, n_items)
            sol_pred,_ = clf_test.solve_model(V_pred)
            sol_true,_ = clf_test.solve_model(V_true)

            regret_full[kn_nr] = max_min_ind *( optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))
            if accuracy:
                cf = confusion_matrix(sol_true, sol_pred,labels=[0,1])
                cf_list.append(cf)
            else:
                cf_list.append(np.zeros((2,2)))
        info['regret_full'] = np.median(regret_full)
        tn, fp, fn, tp = np.sum(np.stack(cf_list),axis=0).ravel()
        info['tn'],info['fp'],info['fn'],info['tp'] =(tn,fp,fn,tp)
        return info

    def validation_score(self, criterion, trch_X_validation=None,trch_y_validation=None,trch_X_test=None,trch_y_test=None,
     knaps_sol_validation=None,knaps_sol_test=None, accuracy=None):
        
        info = dict()
        
        n_items = self.n_items
        solver= self.solver
        optimal_value = self.optimal_value
        param= self.param
        clf_train = self.clf_train
        clf_validation = self.clf_validation
        max_min_ind = 1 if self.maximize else -1

        self.model.eval()
        with torch.no_grad():
            '''
            inputs_train = Variable(trch_X_train)
            target_train = Variable(trch_y_train)
            V_preds_train = self.model(inputs_train)
            info['train_loss'] = criterion(V_preds_train, target_train).item()
            '''
            if trch_X_validation is not None:
                inputs_validation = Variable(trch_X_validation)
                target_validation = Variable(trch_y_validation)
                V_preds_validation = self.model(inputs_validation)
                info['validation_loss'] = criterion(V_preds_validation, target_validation).item()
            
            # if trch_X_test is not None:
            #     inputs_test = Variable(trch_X_test)
            #     target_test = Variable(trch_y_test)
            #     V_preds_test = self.model(inputs_test)
            #     info['test_loss'] = criterion(V_preds_test, target_test).item()

        self.model.train()

        '''
        n_knap_train = len(V_preds_train)//n_items
        regret_full_train = np.zeros(n_knap_train)
        for kn_nr in range(n_knap_train):
            V_true = self.getY(trch_y_train, kn_nr, n_items)
            V_pred = self.getY(V_preds_train, kn_nr, n_items)
            sol_true = knaps_sol_train[kn_nr][0]
            sol_pred,_ = clf.solve_model(V_pred)
            regret_full_train[kn_nr] = max_min_ind *( optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))
        info['train_regret_full'] = np.median(regret_full_train)
        '''
        if trch_X_validation is not None:
            n_knap_validation = len(V_preds_validation)//n_items
            regret_full_validation = np.zeros(n_knap_validation)
            for kn_nr in range(n_knap_validation):
                V_true = self.getY(trch_y_validation, kn_nr, n_items)
                V_pred = self.getY(V_preds_validation, kn_nr, n_items)
                sol_true = knaps_sol_validation[kn_nr][0]
                sol_pred,t = clf_validation.solve_model(V_pred)
                if t is not None:
                    self.time +=t
                regret_full_validation[kn_nr] = max_min_ind *( optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))
            info['validation_regret_full'] = np.median(regret_full_validation)
        # if trch_X_test is not None:
        #     n_knap_test = len(V_preds_test)//n_items
        #     regret_full_test = np.zeros(n_knap_test)
        #     for kn_nr in range(n_knap_test):
        #         V_true = self.getY(trch_y_test, kn_nr, n_items)
        #         V_pred = self.getY(V_preds_test, kn_nr, n_items)
        #         sol_true = knaps_sol_test[kn_nr][0]
        #         sol_pred,_ = clf_test.solve_model(V_pred)
        #         regret_full_test[kn_nr] = max_min_ind *( optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))
        #     info['test_regret_full'] = np.median(regret_full_test)
        return info

    def test_score(self, criterion, trch_X_test=None,trch_y_test=None,knaps_sol_test=None, accuracy=None):
        
        info = dict()
        
        n_items = self.n_items
        solver= self.solver
        optimal_value = self.optimal_value
        param= self.param
        clf_train = self.clf_train
        clf_validation = self.clf_validation
        clf_test = self.clf_test
        max_min_ind = 1 if self.maximize else -1

        self.model.eval()
        with torch.no_grad():
            if trch_X_test is not None:
                inputs_test = Variable(trch_X_test)
                target_test = Variable(trch_y_test)
                V_preds_test = self.model(inputs_test)
                info['test_loss'] = criterion(V_preds_test, target_test).item()

        self.model.train()
        if trch_X_test is not None:
            n_knap_test = len(V_preds_test)//n_items
            regret_full_test = np.zeros(n_knap_test)
            for kn_nr in range(n_knap_test):
                V_true = self.getY(trch_y_test, kn_nr, n_items)
                V_pred = self.getY(V_preds_test, kn_nr, n_items)
                sol_true = knaps_sol_test[kn_nr][0]
                sol_pred,_ = clf_test.solve_model(V_pred)
                regret_full_test[kn_nr] = max_min_ind *( optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))
            info['test_regret_full'] = np.median(regret_full_test)
        return info        


    def fit(self, x_train, y_train,x_validation=None,y_validation=None,x_test=None,y_test=None):
        start = time.time()
        qids = np.array(x_train[:,0], dtype=int) # qid column
        x_train = x_train[:,1:] # without group ID
        validation = (x_validation is not None) and (y_validation is not None)
        test = (x_test is not None) and (y_test is not None)
        if self.early_stopping:
            validation_rslt =[]

        accuracy_measure = self.accuracy_measure
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
        else:
            trch_X_validation = None
            trch_y_validation = None

        if test:
            x_test = x_test[:,1:]
            if self.doScale:
                x_test = self.scaler.transform(x_test)
            trch_X_test = torch.from_numpy(x_test).float()
            trch_y_test = torch.from_numpy(np.array([y_test]).T).float()
        else:
            trch_X_test = None
            trch_y_test = None
        
        # basics
        n_items = self.n_items
        n_knapsacks = len(trch_X_train)//n_items
        param= self.param
        #solver_train= self.solver_train
        #solver_test = self.solver_test
        solver = self.solver
        relax= self.relax
        validation_relax= self.validation_relax

        optimal_value = self.optimal_value
        max_min_ind = 1 if self.maximize else -1
        clf_train =  solver(relax=relax,method=self.method,#obj_cut= self.obj_cut,
            reset=self.reset,presolve= self.presolve,warmstart = self.warmstart, **param)
        clf_train.make_model()
        self.clf_train = clf_train
        if validation:
            clf_validation =  solver(relax=validation_relax,method=self.method,#obj_cut=self.obj_cut,
                reset=self.reset,presolve= self.presolve,warmstart = self.warmstart, **param)
            clf_validation.make_model()
            self.clf_validation = clf_validation
        if test:
            clf_test =  solver(relax=self.test_relax,method=self.method,#obj_cut= self.obj_cut,
                reset=True,presolve= False, **param)
            clf_test.make_model()
            self.clf_test = clf_test

        # prepping
        knaps_V_true = [self.getY(trch_y_train, kn_nr, n_items) for kn_nr in range(n_knapsacks)]
        #self.logger.info('Training Initiation ! Time:%s\n' %str(datetime.datetime.now()) )
        logging.info('Training Initiation ! Time:%s\n' %str(datetime.datetime.now()))
        knaps_sol = [None for V_true in knaps_V_true]
        #knaps_sol = [clf_train.solve_model(V_true,scheduling=True) for V_true in knaps_V_true]
        #self.logger.info('Solving Training instances Completed ! Time:%s\n' %str(datetime.datetime.now()) )
        
        #for k in knaps_sol:
        #    self.time+=k[2]
        validation_time = 0
        test_time = 0
        training_time = 0
        if validation:
            n_knapsacks_validation = len(trch_X_validation)//n_items
            knaps_V_true_validation = [self.getY(trch_y_validation, kn_nr, n_items) for kn_nr in range(n_knapsacks_validation)]
            logging.info('Solving Validation instances ' )
            validation_start = time.time()
            knaps_sol_validation = [clf_validation.solve_model(V_true) for V_true in knaps_V_true_validation]
            validation_end = time.time()
            validation_time += validation_end - validation_start
            #knaps_sol_validation = None
            logging.info('Solving Validation instances Completed ! Time:%s\n' %str(datetime.datetime.now()) )
            for k in knaps_sol_validation:
                if k[1]:
                    self.time+=k[1]
        #self.logger.info('Solving Validation instances Completed ! Time:%s\n' %str(datetime.datetime.now()) )
        
        if test:
            n_knapsacks_test = len(trch_X_test)//n_items
            knaps_V_true_test = [self.getY(trch_y_test, kn_nr, n_items) for kn_nr in range(n_knapsacks_test)]
            logging.info('Solving Test instances ' )
            test_start = time.time()
            knaps_sol_test = [clf_test.solve_model(V_true) for V_true in knaps_V_true_test]
            #knaps_sol_test = None
            test_end = time.time()
            logging.info('Solving Test instances Completed ! Time:%s\n' %str(datetime.datetime.now()) )
            test_time+= test_end - test_start


        #self.logger.info('Solving Test instances Completed ! Time:%s\n' %str(datetime.datetime.now()) )
        


        
        # network
        if not self.model:
            self.model = LinearRegression(trch_X_train.shape[1],1) # input dim, output dim
        # loss
        criterion = nn.MSELoss()
        self.criterion = criterion
        optimizer = self.optimizer(self.model.parameters(), **self.hyperparam)
        num_epochs = self.epochs

        # training
        subepoch = 0 # for logging and nice curves
        logger = [] # (dict_epoch, dict_train, dict_test)
        test_result = []
        knapsack_nrs = [x for x in range(n_knapsacks)]
        for epoch in range(num_epochs):

            random.shuffle(knapsack_nrs) 
            cnt = 0
            for kn_nr in knapsack_nrs:
                logging.info('Epoch:%d-%d\n' %(epoch,cnt)) 
                if self.timelimit is not None and self.time>= self.timelimit:
                    logging.info("TIMELIMIT Reached!!!!!!!!!")
                    break
                if knaps_sol[kn_nr] is None:
                    V_true = knaps_V_true[kn_nr]
                    logging.info("Solving V_true now!!!!!!!!!")
                    knaps_sol[kn_nr] = clf_train.solve_model(V_true)
                    logging.info(" V_true Solved!!!!!!!!!")
                    if knaps_sol[kn_nr][1]:
                        self.time+= knaps_sol[kn_nr][1]
                    

                    # self.sol_hist.append(knaps_sol[kn_nr][0])
                    # if len(self.sol_hist)>self.obj_cut:
                    #     _= self.sol_hist.pop(0)


                sol_true,_ = knaps_sol[kn_nr]
                
                if (self.timelimit is not None and self.time>= self.timelimit) or len(knaps_sol[kn_nr]) < 1:
                    break                       

                # the true-shifted predictions
                V_pred = self.getYpred(self.model, trch_X_train, kn_nr, n_items)
                V_spo = (2*V_pred - V_true)   
                timelimit = (self.timelimit - self.time) if self.timelimit else GRB.INFINITY             

                sol_spo,t = clf_train.solve_model(V_spo,
                 timelimit = timelimit )
               
        
                end = time.time()
                if t is not None:
                    self.time +=t
                if sol_spo is None:
                    break # timeout reached while solving the above
         
                
                 
                grad = (sol_spo - sol_true)*max_min_ind #*2
                '''
                if self.degree == 2:
                    sol_pred = solver_train(V_pred, **param)
                    reg = (optimal_value(V_true,sol_true,**param) - optimal_value(V_true,sol_pred,**param))* max_min_ind 
                    grad = reg*grad
                '''
                batch_X = self.getTorchData(trch_X_train,kn_nr,n_items)
                batch_Y = self.getTorchData(trch_y_train,kn_nr,n_items)
                train_fwdbwd_grad(self.model, optimizer, batch_X, batch_Y, torch.from_numpy(np.array([grad]).T).float())
                cnt += 1
                subepoch += 1              
                if self.verbose:

                    if cnt % 50 == 0:
                        dict_validation = {}
                        if validation:
                            validation_start = time.time()
                            dict_val = self.validation_score(criterion, trch_X_validation=trch_X_validation,trch_y_validation=trch_y_validation,
                            knaps_sol_validation=knaps_sol_validation, accuracy=accuracy_measure)
                            validation_end = time.time()
                            validation_time += validation_end - validation_start
                            dict_validation['validation_loss'] = dict_val['validation_loss']
                            dict_validation['validation_regret_full'] = dict_val['validation_regret_full']

                            print("Epoch %d::subepoch %d Total time %d, validation time %d & test time %d validation regret:%.2f"%(epoch,
                            subepoch,time.time() - start,validation_time,test_time,dict_validation['validation_regret_full']))
                        if test:
                            test_start = time.time()
                            dict_test = self.test_score(criterion,trch_X_test,trch_y_test,knaps_sol_test,accuracy=accuracy_measure) 
                            test_end  = time.time()
                            test_time += test_end - test_start
                            dict_validation['test_loss'] = dict_test['test_loss']
                            dict_validation['test_regret_full'] = dict_test['test_regret_full']

                        dict_validation["subepoch"] = subepoch
                        dict_validation["Runtime"] = self.time
                        dict_validation["time"] = time.time() - start                        
                        test_result.append(dict_validation)
                        logging.info("Epoch %d::subepoch %d Total time %d, validation time %d & test time %d"%(epoch,subepoch,
                            time.time() - start,validation_time,test_time))
                        ### delete
                       
                        ###
                if self.model_save:
                    if cnt % 10 == 0:
                        logging.info("Model saving:%d-%d\n "%(epoch,subepoch))
                        torch.save(self.model.state_dict(), str(self.model_name+"_Epoch"+str(epoch)+"_"+str(subepoch)+".pth"))


        if self.verbose :
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

