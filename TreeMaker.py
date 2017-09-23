from collections import defaultdict
from treelib import Node, Tree
from math import log2

class NodeData:
    def __init__(self, entropy):
        self.entropy = entropy
        self.majority_class = None
    def isMajority(self, classification):
        return classification == self.majority_class


def makeTree(set, dims, starting_entropy):
    tree = Tree()
    root_node = tree.create_node('Root', 'root', data=NodeData(starting_entropy))
    makeBranch(set, dims, tree, root_node)
    return tree


def makeBranch(set, dims, tree, parent_node):
    #base cases: out of header OR labels are pure
    if checkIfPure(set, dims) or len(dims) == 1: #is 1 instead of 0 because "Class" will be in there
        parent_node.majority_class = calcMajorityClass(set, dims)
        return True
    status = True
    chosen_dim, info_gain, value_label_counts, value_entropies = chooseDecisionDim(set, dims, parent_node.data.entropy)
    #mutate original headers (dimensions) data structure as we will be passing references to children
    dims.remove(chosen_dim)
    #remove the dimension from the dataset
    #set.drop(chosen_dim, 1, inplace=True) #1 is for columns axis


    #set the decision made in tag on parent node
    parent_node.tag = chosen_dim
    #add children
    for value, label_counts in value_label_counts.items():
        subset = set.copy()
        #remove rows from subset which don't match current value for newest decision
        subset.drop(subset[subset[chosen_dim] != value].index, inplace=True)
        value_entropy = value_entropies[value]
        identifier = ''.join([parent_node.identifier, '^', chosen_dim, '=', str(value)])
        print('recursing to node %s with an entropy of %f' % (identifier, value_entropy))
        new_node = tree.create_node(None, identifier, parent=parent_node.identifier, data=NodeData(entropy=value_entropy))
        #RECURSE
        status = status and makeBranch(subset, dims, tree, new_node)
    return status

def calcMajorityClass(set, dims):
    classifications = defaultdict(lambda: 0)
    majority_class = None
    max_instances = -1
    for classification in set[dims[-1]]:
        classifications[classification] += 1
        if classifications[classification] > max_instances:
            max_instances = classifications[classification]
            majority_class = classification
    return majority_class

def checkIfPure(set, dims):
    return len(set[dims[-1]].unique()) < 2

def chooseDecisionDim(set, dims, previous_entropy):
    max_info_gain = -1
    max_info_gain_dim = None
    max_info_value_label_counts = None
    max_info_value_entropies = None
    for dim in dims[0:-1]:
        value_label_counts = getValueLabelCounts(set, dim)
        info_gain, value_entropies = calcInfoGain(set, dim, previous_entropy, value_label_counts)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_dim = dim
            max_info_value_label_counts = value_label_counts
            max_info_value_entropies = value_entropies
    print('max info gain is from dimension: "%s" and is: "%f"' % (max_info_gain_dim, max_info_gain))
    return max_info_gain_dim, max_info_gain, max_info_value_label_counts, max_info_value_entropies

def calcInfoGain(set, dims, previous_entropy, value_label_counts):
    dim_entropy = 0
    value_entropies = {}
    num_rows_total = set.shape[0]
    for value, label_counts in value_label_counts.items():
        num_rows_for_value = calcNumInstances(label_counts)
        weight_factor = num_rows_for_value / num_rows_total
        value_entropies[value] = calcEntropy(label_counts, num_rows_for_value)
        dim_entropy += value_entropies[value] * weight_factor
    return previous_entropy - dim_entropy, value_entropies

def calcNumInstances(label_counts):
    total_instances = 0
    for label, count in label_counts.items():
        total_instances += count
    return total_instances


def calcEntropy(label_count, total_instances):
    entropy = 0
    for count in label_count.values():
        entropy += -1 * count / total_instances * log2(count / total_instances)
    return entropy

def getValueLabelCounts(set, header):
    value_label_counts = defaultdict(lambda: defaultdict(lambda: 0))
    #get count of instances for each label
    for row in set.itertuples():
        value = getattr(row, header)
        label = getLabel(row)
        value_label_counts[value][label] += 1
    return value_label_counts

def getLabel(row):
    return row[-1]

