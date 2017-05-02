from node import Node
import math
import numpy as np
import pylab as pl
import random

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  if not examples: # if example set is empty
      return default
  else:
      N = Node()
      N.value = examples
      N = ID3_execute(N)
  return N

def ID3_execute(N):
    homogenous = check_homogenous(N.value) # check to see if output is homogenous
    if homogenous is not None: # stop iterating on a homogenous node, assign label 
        N.label = homogenous
        return N
    elif not N.value: # if set is empty
        return N
    else:
        # pick best attribute of remaining data
        bestAtt = pick_best_attribute(N.value)
        if bestAtt is False: # if gain ratio is 0, stop iterating and "predict" label
            N.label = mode(N.value)
            return N
        else:
            N.name = bestAtt
            children = split_data(N.value,bestAtt)
            # now recurse through children
            for child in children:
                D = Node()
                D.value = children[child]
                N.children[child] = D
                ID3_execute(N.children[child])
            return N

def test(node, examples):
    ''' 
    test on trained tree
    '''
    accuracy=[]
    if type(examples) is dict: # if single example is a dictionary, turn into list
        examples = [examples]
    for ex in examples: # check each example
        classVal = evaluate(node,ex)
        if classVal == ex['Class']:
            accuracy.append(1)
        else:
            accuracy.append(0)
    return float(sum(accuracy))/float(len(accuracy))

    
def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    if node.label is None:
        # if attribute is missing from example, set classVar to mode instead of evaluating node
        if not example.has_key(node.name):
            classVar = mode(node.value)
        # if example attribute value is not present in the node, use mode of all children instead as evaluator
        elif node.name is None or node.children.get(example[node.name]) is None:
            classVar = mode(node.value)
        else:
            classVar = evaluate(node.children[example[node.name]],example)
    else:
        classVar = node.label
    return classVar

def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    root = node
    prune_dfs(root,node,examples)

def prune_dfs(root,node,examples):
    
    for child in node.children:
        prune_dfs(root,node.children[child],examples)
        if node.label is None:
            rootAcc = test(root,examples)
            node.label = mode(node.value)
            acc = test(root,examples)
            if acc >= rootAcc:
                rootAcc = acc
            else:
                node.label = None

def pick_best_attribute(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. 
            If gain ratio of all the attributes is 0, then return False
    ========================================================================================================
    Output: best attribute
    ========================================================================================================
    '''
    # Your code here
    GR = 0
    attributes = data_set[0].keys()
    Hprior = entropy([i['Class'] for i in data_set])
    for i in attributes:
        if i == 'Class':
            pass
        else:
            g = gain_ratio(data_set,Hprior,i)
            if g > GR:
                GR = g
                att = i
    if GR==0:
        att = False
    return att

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    entDict = dict()
    N = float(len(data_set))
    E = 0
    for i in data_set:
        if not entDict.has_key(i):
            entDict[i] = 1
        else:
            entDict[i] += 1

    for i in entDict:
        if len(entDict)==1:
            E = 0
            break
        else:
            P = entDict[i]/N
            E += -P*math.log(P,2)
    
    return E

def gain_ratio(data_set, Hprior,attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, entropy of Class, attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    # Your code here
    data_dict = split_data(data_set, attribute)
    uniqueVals = data_dict.keys()
    Hposterior = 0
    IV = 0
    if len(uniqueVals) == 1:
        gain_ratio = 0
    else:
        for v in uniqueVals:
            split = [i['Class'] for i in data_set if i[attribute]==v]
            P = float(len(split))/float(len(data_set))
            Hposterior += entropy(split)*P
            IV += P*math.log(P,2)
        gain_ratio = (Hprior - Hposterior)/-IV
    return gain_ratio

def split_data(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    # Your code here
    data_dict = dict()
    for vals in data_set:
        if not data_dict.has_key(vals[attribute]):
            data_dict[vals[attribute]] = [vals]
        else:
            data_dict[vals[attribute]].append(vals)
    return data_dict

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of Class
    ========================================================================================================
    Output: mode of class
    ========================================================================================================
    '''
    # Your code here
    count = dict()
    for i in data_set:
        if not count.has_key(i['Class']):
            count[i['Class']] = 1
        else: 
            count[i['Class']] += 1
            
    m = max(count,key=count.get)
    return m   

def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    # Your code here
    homogenous = 0
    for i in data_set:
        if i['Class'] != data_set[0]['Class']:
            homogenous = None
            break
    
    if homogenous == 0:
        homogenous = data_set[0]['Class']
    return homogenous

def random_subsample(data,N):
    ''' 
    randomly subsample from whole dataset to split into N/2 training, N/2 validation and total-N test
    input:
        data = dataset list of dictionaries
        N = number to subsample
    output:
        training set of N/2 size
        validation set of N/2 size
        test set of len(data) - N
    '''
    random.shuffle(data)
    validate = data[:N/2]
    train = data[N/2:N]
    test = data[N:]
    return train,test,validate

def vary_test(data,range1,range2,shuffleN):
    '''
    train on subset of data from size range1 to range2, shuffling shuffleN times each step
    input: dataset, ranges for stepping, shuffle amount per step
    output: ranges vector, accuracy for unpruned, accuracy for pruned
    '''
    acc_unpruned = np.zeros((shuffleN,range2-range1+1))
    acc_pruned = np.zeros((shuffleN,range2-range1+1))
    xAxis = range(range1,range2+1)
    for N in xAxis:
        for sh in range(0,shuffleN):
            train_data,test_data,validate_data = random_subsample(data,N)
            Tree = ID3(train_data,0)
            prune(Tree,validate_data)
            acc_pruned[sh,N-range1] = test(Tree,test_data)
            Tree = ID3(train_data+validate_data,0)
            acc_unpruned[sh,N-range1] = test(Tree,test_data)
        print(N)
            
    return xAxis,acc_unpruned,acc_pruned

def plot_results(xAxis,acc_unpruned,acc_pruned):
    '''
    plot results from np.array input size mxn, where m = shuffled samples and n = training size steps
    '''
    meanAcc_pruned = np.mean(acc_pruned,axis=0)
    meanAcc_unpruned = np.mean(acc_unpruned,axis=0)
    pl.plot(xAxis,meanAcc_unpruned,label='unpruned')
    pl.plot(xAxis,meanAcc_pruned,label='pruned')
    pl.xlabel('training size')
    pl.ylabel('accuracy on test')
    pl.title('Test accuracy over training size')
    pl.legend(loc='lower right')
    pl.show()


# def breadth_first_search(root):
#     '''
#     given the root node, will complete a breadth-first-search on the tree, returning the value of each node in the correct order
#     '''
#     # check if the root is null
#     if root == None:
#         return
#     queue = []
#     # queue is not empty,push the children of this node into the queue
#     queue.append(root)
#     # check if queue is empty
#     while (queue):
#         children = queue[0].children
#         if children != None:
#             for key in children:
#                 queue.append(children[key])
#         # res=res+str(queue.pop(0).get_value())+' '
#         Node = queue.pop(0)
#         if (Node.label == None):
#             print Node.name
#         else:
#             print Node.label#, Node.leaf_data, len(Node.leaf_data)