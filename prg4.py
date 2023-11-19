from pprint import pprint
import pandas as pd
from pandas import DataFrame

# df_tennis = DataFrame.from
df_tennis = pd.read_csv('p4.csv')

# Calculate the Entropy of given probability
def entropy(probs):
    import math
    return sum([-prob * math.log(prob, 2) for prob in probs])

def entropy_of_list(a_list):  # Entropy calculation of list of discrete val ues(YES / NO)
    from collections import Counter
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list) * 1.0
    probs = [x / num_instances for x in cnt.values()]
    return entropy(probs)  # Call Entropy:

# The initial entropy of the YES/NO attribute for our dataset.
# print(df_tennis['PlayTennis'])
total_entropy = entropy_of_list(df_tennis['Play Tennis'])
print("Entropy of given PlayTennis Data Set:", total_entropy)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    '''
 Takes a DataFrame of attributes,and quantifies the entropy of a target
 attribute after performing a split along the values of another attribute.
 '''  # print(df_split.groups)

    # Calculate Entropy for Target Attribute, as well as
    # Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
    # print("NOBS",nobs)
    df_agg_ent = df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x) / nobs]})[
        target_attribute_name]
    # print("FAGGED",df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    # if traced: # helps understand what fxn is doing:
    # Calculate Information Gain:
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy

print('Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'Play Tennis')))
print('Info-gain for Humidity is: ' + str( information_gain(df_tennis,'Humidity', 'Play Tennis')))
print('Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'Play Tennis')))
print('Info-gain for Temperature is:' + str( information_gain(df_tennis, 'Temperature','Play Tennis')))

def id3(df, target_attribute_name, attribute_names, default_class=None):  # Tally target attribute
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])  # class of YES /NO
    # First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return next(iter(cnt))
    # Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class
        # Otherwise: This dataset is ready to be divvied up!
    else:
        # [index_of_max] # most common value  of  target  attribute in dataset,  
        default_class = max(cnt.keys())
        # Choose Best Attribute to split on:
        gainz = [information_gain(df, attr, target_attribute_name)
                 for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        # Create an empty tree, to be populated in a moment
        tree = {best_attr: {}}
        remaining_attribute_names = [
            i for i in attribute_names if i != best_attr]
    
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute_name,remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree
        return tree

# Predicting Attributes
attribute_names = list(df_tennis.columns)
attribute_names.remove('Play Tennis')  # Remove the class attribute
print("Predicting Attributes:", attribute_names)

# Tree Construction

tree = id3(df_tennis, 'Play Tennis', attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)

def test(query, tree, default=None):
    attribute = next(iter(tree))
    if query[attribute] not in tree[attribute].keys():
        return default
    result = tree[attribute][query[attribute]]
    return func(query, result) if isinstance(result, dict) else result
query = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}
ans = test(query, tree)
print('Can tennis be played? : '+ans)