# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:20:53 2018

@author: edemirovic
"""

#1) Pick a number of elements per group (let's say 48 like we did before).
#2) Pick some function to generate the profits (some polynomial or a composition of polynomials, something). Would be great if the weight had some influence on the profits, perhaps its one of the parameters, but it shouldn't be the dominating factor.
#3) Generate groups according to the above function.
#4) Generate one set of weights (say 48) and apply these to each group.
#5) For each group, apply some transformation. I am thinking the easier would be to pick a value for each group, and multiply/divide each profit with that value. This multiplication/division value would be different for each group. This was we would capture the importance of grouping items together, so there's some meaning to looking at the items together. It would also make regression not appropriate, or any method that cannot reason on a group level. I am curious if there are other transformations we could consider. Multiplication is convenient since it would preserve the profit/weight ratio (I guess this is important, makes learning easier).

#How to solve:

#1) We'd train on the profit / weight ratio for the training data.
#2) For test data, we'd rank the items, and apply a greedy algorithm.



#based on some parameters, generate lots of values
#within a group, I will have three batches of 16 items. Each batch has a weight either 3, 5, or 7.

#first step is: generate heaps of profit values. What is the function?

import math
import random


#returns an array where the i-th element is [n, x, y], where n = f(x, y)
def generate_heaps_of_numbers_and_attributes():
    #generate a bunch of numbers
    values = [[int(math.sin(math.radians(i)) * math.sin(math.radians(j)) * 1000), i, j] for i in range(361) for j in range(361)]
    
    #shift the values to the right so that 0 is the smallest
    min_val = min(values, key=lambda x: x[0])[0]
    if min_val < 0:
        for i in range(len(values)):
            values[i][0] += abs(min_val)
    
    values.sort(reverse=True)
 
    #create a new list which contains only unique values        
    unique_list = []
    seen_values = set()
    for i in range(len(values)):
        if values[i][0] in seen_values:
            continue
        unique_list.append(values[i])
        seen_values.add(values[i][0])
        
    return unique_list

#array is a list where the i-th element is [value, x, y, weight]. 
def apply_order_preserving_transformation(array, transformation_type):
    #return array

    assert(transformation_type == "multiply") #this is the only one I am allowing for now, I need to do more complicated transformation for the other ones since the goal is to preserve the RATIO orderings, not the profit orderings

    #exponential -> convert into profitability, then square, then return, but it will skew the values

    if transformation_type == "plus":
        min_val = min(array, key=lambda x: x[0])[0]
        max_val = max(array, key=lambda x: x[0])[0]        
        c = random.randint(min_val, max_val)   
        return [ [x[0] + c, x[1], x[2], x[3]] for x in array] 
    if transformation_type == "multiply":
        c = random.randint(2, 10)
        return [ [x[0]*c, x[1], x[2], x[3]] for x in array]
    if transformation_type == "square":
        return [ [x[0]*x[0], x[1], x[2], x[3]] for x in array]
    else:
        assert(1==2) #unknown transformation
        
        
#returns an array a where len(a) % k == 0
#this done by trimming the last values of array
def make_size_multiple_of_k(array, k):
    m = int(len(array) / k)
    a = [array[i] for i in range(m*k)]
    assert(len(a) % k == 0)  
    return a

def partition_into_groups(array, group_size):
    assert(len(array) % group_size == 0)
    return [ [array[group_size*i + j] for j in range(group_size)] for i in range(int(len(array)/group_size)) ]
    
def remove_and_return_random_element(array):
    assert(len(array) > 0)
    old_size = len(array)
    
    i = random.randint(0, len(array)-1)
    val = array[i]
    del array[i]
    
    assert(len(array) == old_size - 1)
    return val


def shuffle_around_items(benchmark):
    #divide the benchmark into three parts, with each part has the same weight
    w1 = benchmark[:16]
    w2 = benchmark[16:32]
    w3 = benchmark[32:]
    
    for i in range(15):
        assert(w1[i][3] == w1[i+1][3])
        assert(w2[i][3] == w2[i+1][3])
        assert(w3[i][3] == w3[i+1][3])

    random.shuffle(w1)
    random.shuffle(w2)
    random.shuffle(w3)
    
    return w1 + w2 + w3

def generate_benchmarks(n_benchmarks, apply_transformations):
    group_size = 48
    
    v = generate_heaps_of_numbers_and_attributes()
    v = make_size_multiple_of_k(v, group_size)
    
    #v[i] = [profit, attribute1, attribute2], such that profit = f(attribute1, attribute2)
    #therefore v[i] represents a knapsack item
    #now let's assign weights to these items
    #in our case, we use three weights [3, 5, 7]
    #we will enforce fairness so that the values are equality distributed across weights
    
    weights = [3, 5, 7]
    counter = 0    
    for i in range(len(v)):
        v[i].append(weights[counter])
        counter += 1
        
        if counter == len(weights):
            counter = 0
            random.shuffle(weights)

    weights.sort()            
       
    #each knapsack benchmark will have 'group size' elements
    #each weight will be equally represented in a benchmark
    #in additon, we want to make sure that low/high values are also equally represented for a given weight in each benchmark
    #therefore, let k = 'group size' / len(weights)
    #for a weight w, we will partition all items with weight w into k buckets, where the profits of items in bucket i are always greater than the profit of items in bucket i+1
    
    #bucket[i][j] - j-th bucket for i-th weight
    buckets = []    
    assert(group_size % len(weights) == 0)
    k = int(group_size / len(weights))
    for w_i in range(len(weights)):
        w = weights[w_i]
        items = [x for x in v if x[3] == w]
        items = make_size_multiple_of_k(items, k)
        bucket_size = int(len(items) / k)
        items = partition_into_groups(items, bucket_size)
        buckets.append(items)
    
   
    #this is a check that doesn't necessarily always hold but its simpler if it does
    s = len(buckets[0][0])
    for i in buckets:
        for j in i:
            assert(s == len(j))
            
    import copy
    original_buckets = [copy.deepcopy(buckets[i]) for i in range(len(buckets))]
    
    benchmarks = []
    for i in range(n_benchmarks):
        
        bench = []
        #for each benchmark, given a weight w, there will be one item for each bucket[i]
        for w_i in range(len(weights)):
            #debug
            for j in range(k):
                assert(len(buckets[w_i][j]) == len(buckets[w_i][0]))
            
            assert(k == len(buckets[w_i]))
            for j in range(k):                
                item = remove_and_return_random_element(buckets[w_i][j])
                bench.append(item)


            #debug    
            for j in range(k):
                assert(len(buckets[w_i][j]) == len(buckets[w_i][0]))     
        
        s = len(buckets[0][0])
        for m in buckets:
            for v in m:
                assert(s == len(v))

        
        assert(len(bench) == group_size)
        benchmarks.append(bench)
        
        #it's likely that we'll run out of items before we create all our benchmarks, so refill the buckets when they are empty
        if len(buckets[0][0]) == 0:
            print("copied")
            buckets = [copy.deepcopy(original_buckets[i]) for i in range(len(original_buckets))]

        
        
    #now multiply the profits by their weights
    for b_i in range(len(benchmarks)):
        b = benchmarks[b_i]
        for i in range(len(b)):
            b[i][0] *= b[i][3]

    #apply the transformations
    transformations = ["multiply"] # ["plus", "multiply", "square"]
    transformed_benchmarks = []
    counter = 0
    for i in range(len(benchmarks)):
        b = benchmarks[i]
        
        if apply_transformations:
            b = apply_order_preserving_transformation(benchmarks[i], transformations[counter])
        
        b = shuffle_around_items(b)
        
        counter += 1
        counter %= len(transformations)
        transformed_benchmarks.append(b)
        
    return transformed_benchmarks


def get_artificial_data(n=100, unweighted=False, apply_transformations=False):
    benchmarks = generate_benchmarks(n, apply_transformations)
    group_size = len(benchmarks[0])
    assert(len([benchmarks[i] for i in range(n) if len(benchmarks[i]) != group_size]) == 0) #all groups should be of the same size
     
    X = [[i] + benchmarks[i][j][1:3] for i in range(n) for j in range(group_size)]
    Y = []
    weights = [benchmarks[0][i][3] for i in range(group_size)]
    
    if unweighted == True:
        Y = [benchmarks[i][j][0]/weights[j] for i in range(n) for j in range(group_size)]
        weights = [1 for i in range(group_size)]
    else:
        Y = [benchmarks[i][j][0] for i in range(n) for j in range(group_size)]
        weights = [benchmarks[0][i][3] for i in range(group_size)]
      
    #debug - verify that the weight pattern is the same for all groups
    if unweighted==False:
        for i in benchmarks:
            assert(len(weights) == len(i)) 
            assert(len(i) == group_size)
            
            for j in range(len(weights)):
                assert(i[j][3] == weights[j])
        
    splitter = int(0.7*n)
    
    import numpy as np
    
    X_train = np.asarray(X[:group_size*splitter])
    X_test  = np.asarray(X[group_size*splitter:])
    
    Y_train = np.asarray(Y[:group_size*splitter])
    Y_test  = np.asarray(Y[group_size*splitter:]) 

    import numpy
    weights = numpy.asarray(weights)

    return (X_train, Y_train, X_test, Y_test, weights)