from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import torch
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
from collections import defaultdict
import sys
sys.path.insert(0, '../EnergyCost/')
import logging
from sgd_learner import *
from EnergyCost.ICON import *
import random
import time,datetime
#from EnergyCost.ICON import*
# adapted from https://github.com/bwilder0/aaai_melding_code 


class qptl_ICON:
    def __init__(self,param,tau=20000,doScale= True,n_items=48,epochs=1,
        net=LinearRegression,verbose=False,validation_relax=True,test_relax = False,
        validation=False,optimizer=optim.Adam,model_save=False,model_name=None, **hyperparams):
        self.n_items = n_items
        self.epochs = epochs
        self.net = net
        self.tau = tau
        self.validation_relax = validation_relax
        self.test_relax= test_relax

        self.hyperparams = hyperparams
        self.verbose = verbose
        self.validation = validation
        self.doScale = doScale
        self.param = param
        self.optimizer = optimizer
        self.model_time = 0.
        self.model_save =  model_save
        self.model_name = model_name
    
        
    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):
        def make_model_matrix(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
            # nbMachines: number of machine
            # nbTasks: number of task
            # nb resources: number of resources
            # MC[m][r] resource capacity of machine m for resource r 
            # U[f][r] resource use of task f for resource r
            # D[f] duration of tasks f
            # E[f] earliest start of task f
            # L[f] latest end of task f
            # P[f] power use of tasks f
            # idle[m] idle cost of server m
            # up[m] startup cost of server m
            # down[m] shut-down cost of server m
            # q time resolution
            # timelimit in seconds
            Machines = range(nbMachines)
            Tasks = range(nbTasks)
            Resources = range(nbResources)
            N = 1440//q

            ### G and h
            G = torch.zeros((nbMachines*N,nbTasks*nbMachines*N))
            h = torch.zeros(nbMachines*N)
            F = torch.zeros((N,nbTasks*nbMachines*N))
            for m in Machines:
                for t in range(N):
                    h[m*N+t] = MC[m][0]
                    for f in Tasks:
                        c_index = (f*nbMachines+m)*N 
                        G[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] =1
                        F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]
            ### A and b
            A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
            A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
            A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N))

            for f in Tasks:
                A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
                A2 [f,(f*N*nbMachines):(f*N*nbMachines + E[f]) ] = 1
                A3 [f,(f*N*nbMachines+L[f]-D[f]+1):((f+1)*N*nbMachines) ] = 1
            b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
            A = torch.cat((A1,A2,A3))    
            return A,b,G,h,torch.transpose(F, 0, 1)

        ############################    
        logging.info('Model Training Starts\n' )
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

        n_items = self.n_items
        epochs = self.epochs
        net = self.net
        param = self.param
        hyperparams = self.hyperparams
        validation_relax = self.validation_relax
        test_relax = self.test_relax
        if validation:
            start_validation = time.time()
            solver_validation  = Gurobi_ICON(relax=validation_relax,reset= True,presolve= True, **param)
            solver_validation.make_model()
            self.solver_validation = solver_validation
            end_validation = time.time()
            validation_time += end_validation -start_validation
        if test:
            start_test = time.time()
            solver_test  = Gurobi_ICON( **param,reset= True,presolve= True)
            solver_test.make_model()
            self.solver_test = solver_test
            end_test = time.time()
            test_time+=  end_test - start_test
        #sol_train = self.solution_func(y)
        if self.validation:
            if validation:
                sol_validation = self.solution_func(y_validation,solver=self.solver_validation)
            if test:
                sol_test = self.solution_func(y_test,solver= self.solver_test)
        
        
        A,b,G,h,F = make_model_matrix(**param)
        #Q= torch.diagflat(torch.ones(F.shape[0])/tau)
        #print(F.shape)
        Q = torch.eye(F.shape[0])/tau
        
        self.Q = Q
        self.G= G
        self.h = h
        self.A = A
        self.b = b
        self.F = F
        
        model = net(X.shape[1],1)
        
        #optimizer = torch.optim.Adam(model.parameters(),**hyperparams)
        optimizer = self.optimizer(model.parameters(), **hyperparams)
        model_params_quad = make_gurobi_model(G.detach().numpy(),h.detach().numpy(),A.detach().numpy(), 
                                              b.detach().numpy(), Q.detach().numpy())
        self.gurobi_model = model_params_quad
        n_knapsacks = X.shape[0]//n_items

        loss_list =[]
        regret_list = []
        
        subepoch= 0
        logger = []
        test_list = []
        n_train = 1 
        for e in range(epochs):
            logging.info('Epoch %d'%e )

            subepoch_list = [j for j in range(n_knapsacks)]
            random.shuffle(subepoch_list) 

            for i in range(n_knapsacks):
                n_start =  n_items*subepoch_list[i]
                n_stop = n_start + n_items
                c_true = torch.mm(F,torch.tensor(y[n_start:n_stop],dtype=torch.float ).unsqueeze(1))
                X_tensor = torch.tensor(X[n_start:n_stop,:],dtype=torch.float)
                c_pred = (model(X_tensor))
                
                c_pred  = torch.mm(F,model(X_tensor))
                #logging.info('Call to qp function starts' )
                
                solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)
                x = solver(Q.expand(n_train, *Q.shape),
                c_pred.squeeze(), G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), 
                A.expand(n_train, *A.shape), b.expand(n_train, *b.shape))
                #logging.info('Call to qp function ends' )
                
                self.model_time +=solver.Runtime()
                loss = (x.squeeze()*c_true.squeeze()).mean()
                optimizer.zero_grad()
                #print(loss)
                loss.backward()
                optimizer.step()
                self.model = model
                subepoch += 1
                if self.model_save:
                    if i%10==0:
                        logging.info("Model saving:%d-%d\n "%(e,i))
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(i)+".pth"))

                if i%50==0:
                    if self.verbose:
                        #train_result = self.test(X,y,sol_train)
                        dict_validation = {}
                        logging.info('Validation starts\n ' )
                        if validation:
                            start_validation = time.time()
                            validation_result = self.test(X_validation,y_validation,sol_validation,solver= self.solver_validation)
                            logging.info('Validation on test data starts\n ' )
                            self.model_time+= validation_result[3]
                            dict_validation['validation_regret']=validation_result[0]
                            dict_validation['validation_mse'] = validation_result[1]
                            dict_validation['validation_loss']=validation_result[2]
                            end_validation = time.time()
                            validation_time += end_validation -start_validation

                        if test:
                            start_test = time.time()
                            test_result = self.test(X_test,y_test,sol_test,solver= self.solver_test)
                            logging.info('Validation Ends \n ' )
                            dict_validation['test_regret'] = test_result[0]
                            dict_validation['test_mse'] =test_result[1]
                            dict_validation['test_loss'] = test_result[2]
                            end_test = time.time()
                            test_time+=  end_test - start_test

                        dict_validation['subepoch'] = subepoch
                        dict_validation['Runtime'] = self.model_time
                        dict_validation['time'] = time.time() - start  
                        
                        test_list.append(dict_validation)
                        logging.info("Epoch %d::subepoch %d Total time %d, validation time %d & test time %d"%(e,
                            i,time.time() - start,validation_time,test_time))
            
            print('Epoch[%d::%d], loss(train):%.2f at %s'%(e+1, i, loss,datetime.datetime.now() ))

                    
                        
        
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
    def solution_func(self,y,solver,**params):
        n_items = self.n_items
        n_knap = len(y)//n_items
        regret_full = np.zeros(n_knap)
        solution = []
        for kn_nr in range(n_knap):
            n_start  = kn_nr * n_items
            n_stop  = n_start + n_items
            V = y[n_start:n_stop]
            solution.append(solver.solve_model(V))
            #print(solver.solve_model(V))

        return solution
    
    def test(self,X,y,sol,solver):
        Q= self.Q
        G = self.G
        h =  self.h
        A = self.A
        b =  self.b
        F = self.F
        model_params_quad = self.gurobi_model 
        time = 0
        n_train = 1 
        n_items = self.n_items
        epochs = self.epochs
        net = self.net
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X,dtype=torch.float)
        c_pred = (model(X_tensor))
        y_pred = c_pred.detach().numpy().squeeze()
        model.train()
        n_knap = len(y)//n_items
        sol_pred = self.solution_func(y_pred,solver=solver)
        regret_list= []
       
        loss_list = []
        for i in range(n_knap):
            
            n_start =  n_items*i
            n_stop = n_start + n_items
            
            regret = (y[n_start:n_stop]*(sol_pred[i][0] - sol[i][0])).sum()
            ### to be delted
            #print(regret)
            #print(sol_pred[i][0])
            #print(sol[i])
            ###########
            regret_list.append(regret)
            
            c  = torch.mm(F,c_pred[n_start:n_stop])
            c_true = torch.mm(F,torch.tensor(y[n_start:n_stop],dtype=torch.float ).unsqueeze(1))
            solver = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params_quad)
            x = solver(Q.expand(n_train, *Q.shape),
                c.squeeze(), G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
             
            loss = (x.squeeze()*c_true.squeeze()).mean().item()
            loss_list.append(loss)
            time += solver.Runtime()
            
            

        return np.median(regret_list), mse(y,y_pred), np.median(loss_list),time






    
