# Import relevant libraries
import numpy as np
import pandas as pd
from pprint import pprint
from math import log

# Import training data
# The training data is located in the dataset folder (KDD dataset)
train_data = pd.read_csv('ids-train.txt',sep='\s+',header=None)
train_data.columns = ['duration', 'protocol_type', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'class']
#print(train_data)

# Import the testing dataset 
# It is located in the dataset directory
test_data = pd.read_csv('ids-test.txt',sep='\s+',header=None)
test_data.columns = ['duration', 'protocol_type', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'class']


# Calculate the entropy of the data
def calculate_entropy(data):
    
    num_counts = len(data)  
    label_dict = {} 
    for featVec  in data:
        currentLabel = featVec[-1]  
        if currentLabel not in label_dict.keys():  
            label_dict[currentLabel] = 0;
        label_dict[currentLabel] += 1;

    # Find all entropy types
    entropy = 0.0  
    for key in label_dict:
        prob = float(label_dict[key])/num_counts 
        entropy -= prob * log(prob, 2)
    #print(entropy)
    return entropy

eps = np.finfo(float).eps
def relative_entropy(data,attribute):
    Class= data.keys()[-1]  
    #print(Class)
    target_variables = data[Class].unique()  #TDecision
    variables = data[attribute].unique()    
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(data[attribute][data[attribute]==variable][data[Class] == target_variable])
            den = len(data[attribute][data[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(data)
        entropy2 += -fraction2*entropy
        #print(entropy2)
    return abs(entropy2)


def decisiion_Tree(data,tree=None): 
    
    node = best_att(data)
    
    # Values of attribute
    val_attr = np.unique(data[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   
    # Loop for the making tree using recursion
    for sample in val_attr:
        
        subtree = pref_node(data,node,sample)
        Class,counts = np.unique(subtree['class'],return_counts=True)                        
        
        if len(counts)==1:
            tree[node][sample] = Class[0]                                                    
        else:   
        	# recursion     
            tree[node][sample] = decisiion_Tree(subtree) 
                   
    return tree

def best_att(data):
    Entropy_att = []
    Info_Gain = []
    for key in data.keys()[:-1]:       
        Info_Gain.append(calculate_entropy(data)-relative_entropy(data,key))
    return data.keys()[:-1][np.argmax(Info_Gain)]
  
def pref_node(data, node,value):
    return data[data[node] == value].reset_index(drop=True)

def predict(data,tree,default = 1):
    
    for key in list(data.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][data[key]] 
            except:
                return default
 
            result = tree[key][data[key]]
   
            if isinstance(result,dict):
                return predict(data,result)

            else:
                return result
            
# Accuracy of prediction
def test_accuracy(data,tree):
    
    pred = data.iloc[:,:-1].to_dict(orient = "records")
    #print(pred)
    
    
    #Create a empty DataFrame 
    prediction = pd.DataFrame(columns=["Outcome"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        prediction.loc[i,"Outcome"] = predict(pred[i],tree,1.0)
        #print(prediction)
    print('Accuracy of testing is: ',(np.sum(prediction["Outcome"] == data["class"])/len(data)))
    


tree = decisiion_Tree(train_data)
import pprint
pprint.pprint(tree)


# Accuracy of Training
train_true = train_data.iloc[:,:-1].to_dict(orient = "records")
train_pred = pd.DataFrame(columns=["Outcome"])

for i in range(len(train_data)):
        train_pred.loc[i,"Outcome"] = predict(train_true[i],tree,1.0)
        #print(prediction)
print('Accuracy of training is: ',(np.sum(train_pred["Outcome"] == train_data["class"])/len(train_data)))

#test(test_data,tree)
test_accuracy(test_data, tree)
