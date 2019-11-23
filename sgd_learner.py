import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn, optim
from torch.autograd import Variable
from KnapsackSolving import *
from operator import itemgetter
import itertools
from multiprocessing.pool import ThreadPool
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import sys

from EnergyCost.ICON import *
class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)  # input and output is 1 dimension
        

    def forward(self, x):
        out = self.linear(x)
        return out
class GridRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)  # input and output is 1 dimension
        self.relu = nn.ReLU()
        

    def forward(self, x):
        out = self.relu(( self.linear(x)))
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
def shortest_path(V_pred,height=3,width=3):
    import networkx as nx
    V_pred = np.where(V_pred<0,0,V_pred)
    def create_graph(height,width):
        #G = nx.Graph()
        G= nx.DiGraph()
        G.add_nodes_from([str(i)+","+str(j) for i in range(height+1) for j in range(width+1) ])
        return G
    def add_weight(G,L,height,width):
        # G is the directed graph L is the the list of weights
        t = 0
        d = {}
        for i in range(height+1):
            for j in range(width+1):
                if i< width:
                    #G.add_weighted_edges_from([( str(i)+","+str(j),str(i+1)+","+str(j) ,L[t])])
                    G.add_edge(str(i)+","+str(j),str(i+1)+","+str(j), weight=L[t] )
                    d[str(i)+","+str(j),str(i+1)+","+str(j)]= t
                    #d[str(i+1)+","+str(j),str(i)+","+str(j)]= t
                    t+=1
                if j< height:
                    #G.add_weighted_edges_from([( str(i)+","+str(j),str(i)+","+str(j+1) ,L[t])])
                    G.add_edge(str(i)+","+str(j),str(i)+","+str(j+1), weight=L[t] )
                    d[str(i)+","+str(j),str(i)+","+str(j+1)]= t
                    #d[str(i)+","+str(j+1), str(i)+","+str(j)]= t
                    t+=1
        return G,d
    def path_distance(G,path):
        labels = nx.get_edge_attributes(G,'weight')
        dist= 0
        for l in range(len(path)-1):
            dist+= labels[(path[l],path[l+1])]
        return dist
    H = create_graph(height,width)
    H, dt = add_weight(H,V_pred,height,width)
    sp =  nx.bellman_ford_path(H,"0,0",str(height)+","+str(width) )
    #sp =  nx.dijkstra_path (H,"0,0",str(height)+","+str(width) )
    ret = np.zeros(V_pred.shape[0])
    for i in range(len(sp)-1):
        ret[dt[sp[i],sp[i+1]]] =1
    return ret



def get_kn_indicators(V_pred, c, weights=None,use_dp= True,relaxation=False,warmstart=None):
    if weights is None:
        weights = np.ones(V_pred.shape[0])
    if use_dp:
        if relaxation:
            solution = solveKnapsackProblemRelaxation(V_pred,weights,c,warmstart=warmstart)
        else:
            solution = solveKnapsackProblem(V_pred,weights,c,warmstart=warmstart)
        return np.asarray(solution['assignments']),solution['runtime']


    ret = np.zeros(V_pred.shape[0])

    # order by profitability
    V_val = V_pred/weights

    for val in sorted(set(V_val), reverse=True):
        same_val = (V_val == val)
        tot_weight = sum(weights[same_val])
        if c>= tot_weight:
            # all in
            ret[same_val] = 1
            c = c - tot_weight
        
        elif c > 0:
            # equal divide
            fraction = c/tot_weight
            ret[same_val] = fraction
            c = 0
            break
        else:
            break
        """
        elif c>0:
            eligible_weights = ((weights<=c) & (V_val == val)) 
            tot_weight = sum(weights[eligible_weights])
            ret[eligible_weights] = weights[eligible_weights]/tot_weight
            c=0 
            break
        """
        '''    
        for w in sorted(set(weights[same_val]),reverse=True):
            if c >= w:
                same_weights = ((weights==w) & (V_val == val))
                #print(same_weights)
                n = min(len(same_weights[same_weights==True]),int(c/w))
                c -= n*w
                fraction = n/len(same_weights[same_weights==True])
                ret[same_weights] = fraction
            if c<=0:
                break
        '''          
        # penalize negative values, will never be in full solution 
    ret[V_pred <= 0] = 0
    return ret
def get_data(trch,kn_nr,n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items
    return trch[kn_start:kn_stop]
def get_data_ICON(trch,kn_nr,n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items+1
    return trch[kn_start:kn_stop]
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

def get_profits_ICON(trch_y, kn_nr, n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items+1
    return trch_y[kn_start:kn_stop].data.numpy().T[0]

def get_profits_pred_ICON(model, trch_X, kn_nr, n_items):
    kn_start = kn_nr*n_items
    kn_stop = kn_start+n_items+1
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

    
def test_fwd(model, criterion, trch_X, trch_y, n_items, capacity, knaps_sol,weights=None,relaxation=False):
    info = dict()    
    model.eval()
    with torch.no_grad():
        # compute loss on whole dataset
        inputs = Variable(trch_X)
        target = Variable(trch_y)
        V_preds = model(inputs)
        info['loss'] = criterion(V_preds, target).item()
    model.train()
        
    n_knap = len(V_preds)//n_items
    regret_smooth = np.zeros(n_knap)
    regret_full   = np.zeros(n_knap)
    cf_list =[]
    time =0 
    # I should probably just slice the trch_y and preds arrays and feed it like that...
    for kn_nr in range(n_knap):
        V_true = get_profits(trch_y, kn_nr, n_items)
        V_pred = get_profits(V_preds, kn_nr, n_items)
        assignments_pred,t = get_kn_indicators(V_pred, c=capacity, weights=weights,relaxation=relaxation)
        
        assignments_true = knaps_sol[kn_nr][0]
        regret_full[kn_nr] = np.sum(V_true * (assignments_true - assignments_pred ) ) 
        if not relaxation:
            cf = confusion_matrix(assignments_true, assignments_pred,labels=[0,1])
            cf_list.append(cf)
        #sol_true = get_kn_indicators(V_true, capacity, weights=weights)
        #sol_pred = get_kn_indicators(V_pred, capacity, weights=weights)
        #regret_smooth[kn_nr] = sum(V_true*(sol_true - sol_pred))
        #regret_full[kn_nr],cf = regret_knapsack([V_true], [V_pred], weights, capacity,assignments=, relaxation=relaxation)
        
        time+=t
    
    info['nonzero_regrsm'] = sum(regret_smooth != 0)
    info['nonzero_regrfl'] = sum(regret_full != 0)
    #info['regret_smooth'] = np.average(regret_smooth)
    info['regret_full'] = np.median(regret_full)
    #info['confusion_matrix'] = np.sum(np.stack(cf_list),axis=0).ravel()
    if not relaxation:
        tn, fp, fn, tp = np.sum(np.stack(cf_list),axis=0).ravel()
        info['tn'],info['fp'],info['fn'],info['tp'] =(tn,fp,fn,tp)
        info['accuracy'] = (tn+tp)/(tn+tp+fn+fp)
    else:
        info['accuracy'] = None
    info['runtime'] =time
    return info

def diffprof(V_pred, index, newvalue, V_true, c, weights=None,use_dp= True):
    sol = get_kn_indicators(V_pred, c, weights,use_dp)
    """# shortcut for 'remains in' and 'remains out'
    if len(V_pred[sol > 0]) != 0:
        if weights is None:
            weights = np.ones(V_pred.shape[0])
        V_val = V_pred/weights
       
        minval = min(V_val[sol > 0])
        oldvalue = V_val[index]
        print("Min",minval,"Old",oldvalue)
        if oldvalue > minval and newvalue > minval:
            # remains in, no change in 'sol'
            return 0
        elif oldvalue < minval and newvalue < minval:
            # remains out, no change in 'sol'
            return 0
    """
    Vnew = np.array(V_pred)
    Vnew[index] = newvalue
    sol_new = get_kn_indicators(Vnew, c, weights,use_dp)
    return sum(V_true*(sol - sol_new)) # difference in obj
def diffprof_grid(V_pred, index, newvalue, V_true, height,width):
    sol = shortest_path(V_pred,height,width)
    Vnew = np.array(V_pred)
    Vnew[index] = newvalue
    sol_new = shortest_path(Vnew, height,width)
    return sum(V_true*(sol - sol_new)) # difference in obj
def knapsack_value(V,sol,**kw):
    return sum(V*sol)


class grid_search:
    def __init__(self,clf,fixed_parameter,variable_parameter,by,max_epochs=10,n_iter= 10):
        self.clf= clf
        self.fixed_parameter = fixed_parameter
        self.variable_parameter = variable_parameter
        self.by = by
        self.n_iter = n_iter
        self.max_epochs= max_epochs

            
    def fit(self,X_train,y_train,X_val,y_val):
        self.X_train = X_train
        self.y_train = y_train
        by = self.by

        def iterate_values(S):
            keys, values = zip(*S.items())
            L =[]

            for row in itertools.product(*values):
                L.append( dict(zip(keys, row)))
            return L  
        
        def fit_func(kwargs):
            foo = self.clf(**kwargs)
            df = pd.DataFrame()
            for i in range(self.n_iter):
                scr = foo.fit(X_train,y_train,X_val,y_val)
                df = pd.concat([df,scr])
            df = df.groupby(['Epoch'],as_index=False).mean()
            return df[by].min(),df['Epoch'][ df[by].idxmin()]
        
        
        fixed ={}
        for k,v in self.fixed_parameter.items():
            fixed[k] =[v]
        var = self.variable_parameter
        z = {**fixed, **var}
        z['epochs']= [self.max_epochs]
        combinations= iterate_values(z)
        pool = ThreadPool(len(combinations))
        results = pool.map(fit_func, combinations)
        
        pool.close()
        pool.join()
        
        mean_scr = [i[0] for i in results]
        epochs = [i[1] for i in results]
        index=  min(enumerate(mean_scr), key=itemgetter(1))[0]
        params= combinations[ index ]
        params['epochs'] = epochs[index]
        params['early_stopping'] = False
        self.fit_result = {"params": combinations,"score":results,"optimal_parameter":params}
        return dict((k, params[k]) for k in var.keys() )        
    def test_score(self,X_test,y_test):
        X_train = self.X_train
        y_train = self.y_train
        def scr_func(kwargs):
            foo = self.clf(**kwargs)
            foo.fit(X_train,y_train)
            train_scr = foo.test_score(X_train,y_train)
            test_scr = foo.test_score(X_test,y_test)
            return [train_scr['regret'],train_scr['loss'],test_scr['regret'],test_scr['loss']]
        params = self.fit_result['optimal_parameter']
        print("Optimum parameter:",params)
        combinations = [params for i in range(self.n_iter)]
        pool = ThreadPool(self.n_iter)
        results = pool.map(scr_func, combinations)
        mean_rslt =np.mean(np.array(results),axis=0)
        return {'train_regret':mean_rslt[0],'train_loss':mean_rslt[1],
               'test_regret':mean_rslt[2],'test_loss':mean_rslt[3]}



def ICON_solution(y_pred,y_test,relax,presolve,reset,n_items=288,solver= Gurobi_ICON,method=-1,**param):
    clf =  solver(relax=relax,method=method,reset=reset,presolve=presolve, **param)
    clf.make_model()
    sol_hist = []

    n_knap = len(y_pred)//n_items
    result = []
    for kn_nr in range(n_knap):
        kn_start = kn_nr*n_items
        kn_stop = kn_start+n_items
        V = y_pred[kn_start:kn_stop]
        V_test = y_test[kn_start:kn_stop]
        logging.info("Oracle called")
        sol,_ = clf.solve_model(V)
        logging.info("Oracle returned")
        sol_hist.append(sol)
        if len(sol_hist)>50:
            _= sol_hist.pop(0)

        opt = knapsack_value(V_test,sol)            
        result.append({"instance":kn_nr,"optimal_value":opt})
    dd = defaultdict(list)
    for d in result:
        for key, value in d.items():
            dd[key].append(value)
    return dd
