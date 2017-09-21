import pandas as pd
import sys
import pprint as pp
from pathlib import Path
from treelib import Node, Tree
from math import log2
from collections import defaultdict

usage = 'TreeMaker.py /path/to/training/dataset /path/to/validation/dataset /path/to/testing/dataset pruning_factor'

if len(sys.argv) < 5:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 5:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)

training_data_path = Path(sys.argv[1])
validation_data_path = Path(sys.argv[2])
test_data_path = Path(sys.argv[3])
pruning_factor = sys.argv[4]

training_set = pd.read_csv(training_data_path)
validation_set = pd.read_csv(validation_data_path)
test_set = pd.read_csv(test_data_path)

headers = list(training_set.columns.values)

num_tuples = len(training_set[headers[-1]])
num_label = {}

#find initial entropy
for val in training_set[headers[-1]]:
    if val in num_label:
        num_label[val] += 1
    else:
        num_label[val] = 1

probability_label = {}
for key in num_label:
    probability_label[key] = num_label[key] / num_tuples

previous_entropy = 0
for key in probability_label:
    previous_entropy += -1 * probability_label[key] * log2(probability_label[key])
print(previous_entropy)


class NodeData:
    def __init__(self, entropy):
        self.entropy = entropy

tree = Tree()

tree.create_node('Root', 'root', data=NodeData(previous_entropy))

def make_branch(set, dims, tree, parent_node):
    #base cases: out of header OR labeles are pure
    chosen_dim, entropy, value_label_counts = chooseDecisionDim(set, dims, parent_node.data.entropy)
    #mutate original headers (dimensions) data structure as we will be passing references to children
    dims.remove(chosen_dim)
    #remove the dimension from the dataset
    set.drop(chosen_dim, 1, inplace=True) #1 is for columns axis

    #TODO remove rows from set which don't match decion tree down branch to current location including newest decision

    #set the decision made in tag on parent node
    parent_node.tag = chosen_dim;
    #add children
    for value in value_label_counts.keys():
        pp.pprint(value)
        identifier = ''.join([chosen_dim, '=', str(value)])
        tree.create_node(None, identifier, parent=parent_node.identifier, data=NodeData(entropy=entropy))
    #TODO check for base cases
    #TODO recurse

def chooseDecisionDim(set, dims, previous_entropy):
    max_info_gain = -1
    max_info_gain_dim = None
    max_info_value_label_counts = None
    for dim in headers[0:-1]:
        value_label_counts = getValueLabelCounts(set, dim)
        info_gain = calcInfoGain(set, dim, previous_entropy, value_label_counts)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_dim = dim
            max_info_value_label_counts = value_label_counts
    print('max info gain is from dimension: "%s" and is: "%d"' % (max_info_gain_dim, max_info_gain))
    return max_info_gain_dim, max_info_gain, max_info_value_label_counts

def calcInfoGain(set, dim, previous_entropy, value_label_counts):
    dim_entropy = 0
    num_rows_total = set.shape[0]
    for value, label_counts in value_label_counts.items():
        num_rows_for_value = calcNumInstances(label_counts)
        weight_factor = num_rows_for_value / num_rows_total
        dim_entropy += calcEntropy(label_counts, num_rows_for_value) * weight_factor
    return previous_entropy - dim_entropy

def calcNumInstances(label_counts):
    total_instances = 0
    for label, count in label_counts.items():
        total_instances += count
    return total_instances


def calcEntropy(label_count, total_instances):
    entropy = 0
    for label, count in label_count.items():
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


make_branch(training_set, headers, tree, tree.get_node('root'))


#TODO remove
#for dim in header[0:-1]:
#    print('info gain for %s: %s' % (dim, calcInfoGain(training_set, dim, previous_entropy)))
