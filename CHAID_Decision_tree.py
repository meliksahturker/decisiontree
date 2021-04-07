"""
CHAID Decision Tree (Non-Binary) Implementation according to Artificial Intelligence: A Modern Approach book.

Sample code to use:

minimumPurity = 0.5
methods = ['entropy', 'gini']
print('Minimum Purity criteria to stop splitting: ' + str(minimumPurity))
for i, method in enumerate(methods):
    print('Method: ' + method)
    print('Tree built looks like: ')
    t = buildDecisionTree(subset = df, wholeset = df, minimumPurity = minimumPurity, method = method)
    recursive_print(t, depth = 0)
    if i == 0:
        print(5*"\n")

"""

import pandas as pd
import numpy as np

#Method to calculate Entropy for all the dataset since calculating Entropy involves
# - calculating Entropy for all Dataset
# - calculating Entropy for each attribute
def entropy_of_dataset(df): 
    entropy = 0 
    target_variable = df.columns.values[-1]
    
    # for every unique instance in target variable, calculate their fraction and add it to entropy
    for value in df[target_variable].unique(): 
        frac = df[target_variable].value_counts()[value] / df[target_variable].shape[0]
        entropy += -frac * np.log2(frac)
    return entropy


# Calculate Entropy for all attributes in the dataset
def entropy(df):
    
    lst_entropy = []
    target_variable = df.columns.values[-1]
    
    # for every attribute in df calculate attribute
    for attribute in df.columns.values[:-1]:
        values_in_attribute  = df[attribute].unique()    # Unique values in attribute
        values_in_target_variable = df[target_variable].unique() # Unique values in target_variable

        entropy_of_attribute = 0
        
        # starting with calculating for every unique value in attribute
        for variable in values_in_attribute:
            entropy_of_each_unique_value = 0 # Entropy of the unique value of the attribute

            for label in values_in_target_variable:
                num = df[attribute][df[attribute] == variable][df[target_variable] == label].shape[0]
                den = df[attribute][df[attribute] == variable].shape[0]
                
                # calculate the ratio by dividing numerator by denumerator
                frac = num / (den + epsilon) # epsilon protects from 0 divison
                
                # Entropy for single unique value in the attribute
                entropy_of_each_unique_value += -frac * np.log2(frac + epsilon)  # epsilon protects from 0 divison

            frac2 = den / df.shape[0]
            entropy_of_attribute += -frac2 * entropy_of_each_unique_value 

        lst_entropy.append(abs(entropy_of_attribute))
    df_entropy = pd.DataFrame(lst_entropy).T
    df_entropy.columns = df.columns.values[:-1]
    return df_entropy


def information_gain(df):
    df_information_gain = (df.shape[1]-1) * [entropy_of_dataset(df)] - entropy(df)
    return df_information_gain


def gini(df):
    lst_gini = []
    
    # For every attribute
    for column in df.columns.values[:-1]: 
        
        # List of unique values in that attribute
        unique_values_in_attribute = df[column].unique() 
        
        # List of gini index values for columns
        lst_g = []
        
        # List of number of instances in unique_value of attribute
        lst_probs_total_counts = [] 
        
        # For every unique_value in that attribute
        for unique_value in unique_values_in_attribute: 
            
            # Probabilities
            probs = df[df[column] == unique_value][df.columns.values[-1]].value_counts(normalize = True).values
            
            # Number of instance for that unique_value
            total_count_in_prob = np.sum(df[df[column] == unique_value][df.columns.values[-1]].value_counts().values) 
            lst_probs_total_counts.append(total_count_in_prob)
            g = 1
            for prob in probs:
                g-= prob**2
            lst_g.append(g)
            
        # Weighted sum of unique_attributes for final gini index value of the attribute
        gini_of_attribute = np.dot(np.array(lst_probs_total_counts) / df.shape[0], lst_g) 
        lst_gini.append(gini_of_attribute)
        
    df2 = pd.DataFrame(lst_gini).T
    df2.columns = df.columns.values[:-1]
    return df2


# next_attribute_to_split() method returns the attribute with highest purity in the given dataset, according to given method, whether 'entropy', or 'gini'
# Standardization of purity value for 'gini' is done here
def next_attribute_to_split(df, method = 'entropy'): 
    # Returns the attribute with highest information gain in df, as well as its value
    if method == 'entropy': 
        return information_gain(df).T.idxmax().values[0], information_gain(df).T.max().values[0]
    
    # Returns the attribute with the minimum gini index in df, as well as its value
    else: 
        return ((0.5 - gini(df)) / 0.5).T.idxmax().values[0], ((0.5 - gini(df)) / 0.5).T.max().values[0]




def buildDecisionTree(subset, wholeset, tree = None, minimumPurity = 0.2, method = 'entropy'): 

    # subset is the subset given to split further.
    # wholeset is the whole data set
    # tree is the empty set to build tree upon
    # minimumPurity is the Purity value to stop splitting furthermore
    # method is the method to calculate purity. Can be 'entropy' or 'gini'
    
    # target_variable is always the last column in the data
    target_variable = df.columns.values[-1]
    
    node, purity = next_attribute_to_split(subset, method = method)
    unique_values_in_node = wholeset[node].unique()
    
    if tree is None:                    
        tree = []
        tree.append(node)

    # for every unique value in the node, subset of the original data 
    for unique_value_of_attribute in unique_values_in_node:

        subset_of_values_in_node = subset[subset[node] == unique_value_of_attribute].reset_index(drop=True)
        number_of_classes_in_subset = subset_of_values_in_node[target_variable].nunique()
        unique_values_in_subset = sorted(subset_of_values_in_node[target_variable].unique())


        # 1- If the remaining examples have all the same target variable, return it
        if number_of_classes_in_subset == 1:
            tree.append([node, unique_value_of_attribute, unique_values_in_subset[0]])
            #tree[node][unique_value_of_attribute] = unique_values_in_subset[0]

        
        # 2- If there's no remaining examples, return the plurality class of WHOLE_SET
        elif (subset_of_values_in_node.shape[0] == 0):
            tree.append([node, unique_value_of_attribute, wholeset[target_variable].value_counts().idxmax()])
            #tree[node][unique_value_of_attribute] = wholeset[target_variable].value_counts().idxmax()
            
        # 3- If there's no attributes left, return the plurality of remaining samples
        elif (subset_of_values_in_node.shape[1] == 2):
            tree.append([node, unique_value_of_attribute, subset_of_values_in_node[target_variable].value_counts().idxmax()])
            #tree[node][unique_value_of_attribute] = subset_of_values_in_node[target_variable].value_counts().idxmax()
            
        # 4- If there are some positive and negative examples but,
        #    purity is below minimum Purity threshold, don't split further, and return the plurality of the subset    
        elif ((method == 'entropy') & (purity < minimumPurity)) | ((method == 'gini') & (purity < minimumPurity)):
            tree.append([node, unique_value_of_attribute, subset_of_values_in_node[target_variable].value_counts().idxmax()])
            #tree[node][unique_value_of_attribute] = subset_of_values_in_node[target_variable].value_counts().idxmax()
           
            
        # 5- If there are some positive and negative examples but,
        #    purity is not below the minimumPurity threshold,
        #    create the subtree recursively using the subset_of_values_in_node data,
        #    dropping the current attribute/node, otherwise it may result in infinite loop
        else:
            tree.append([node, unique_value_of_attribute, buildDecisionTree(subset = subset_of_values_in_node.drop(node, axis = 1), wholeset = wholeset, tree = None, minimumPurity = minimumPurity, method = method)])
            #tree[node][unique_value_of_attribute] = buildDecisionTree(subset = subset_of_values_in_node.drop(node, axis = 1), wholeset = wholeset, tree = None, minimumPurity = minimumPurity, method = method) #Calling the function recursively 
    

    # if all children of a node has same return value, make it a leaf with that value, handling duplicate leafs
    lst_to_check_duplicates = tree[1:]
    value = lst_to_check_duplicates[0][2]
    values_in_lst_to_check_duplicates = []
    for i in range(len(lst_to_check_duplicates)):
        values_in_lst_to_check_duplicates.append(lst_to_check_duplicates[i][2])
    
    if all(x == values_in_lst_to_check_duplicates[0] for x in values_in_lst_to_check_duplicates):
        tree = value
        
    return tree



# recursive_print methods prints the created tree in recursive calls, in order to enable visual inspection
def recursive_print(t, depth = 0):
    for i, item in enumerate(t):
        
        # Seperate different levels of tree with dash, according to depth of the tree
        if i == 0:
            print(depth*6*"-")
            
        # If the element is string, print it, if it's another nested list, call the function recursively.
        if isinstance(item, str):
            print(depth*"     " + item)
        else:
            recursive_print(item, depth = depth + 1)