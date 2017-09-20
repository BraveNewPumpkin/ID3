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

header = list(training_set.columns.values)

num_tuples = len(training_set[header[-1]])
num_label = {}


for val in training_set[header[-1]]:
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
    def __init__(self, entropy, header):
        self.entropy = entropy
        self.header = header

tree = Tree()

tree.create_node('Root', 'root', data = NodeData(previous_entropy, None))

def make_branch(set, headers, tree, parent_node):
    #base cases: out of header OR labeles are pure
    for dim in headers[0:-1]: #TODO: 0:-2?
        value_label_counts = getValueLabelCounts(set, dim)
        header_entropy = 0
        for value, label_count in value_label_counts:
            header_entropy += calcEntropy(label_count) #TODO *weight_factor
        pass

def calcEntropy(label_count):
    entropy = 0
    total_instances = 0
    for label, count in label_count.items():
        total_instances += count
    for label, count in label_count.items():
        entropy += -1 * count / total_instances * log2(count / total_instances)
    return entropy

def getValueLabelCounts(set, header):
    label_counts = defaultdict(lambda: defaultdict(lambda: 0))
    #get count of instances for each label
    for tuple in set.itertuples(index=True):
        value = getattr(tuple, header)
        label = getLabel(tuple)
        label_counts[value][label] += 1
    return label_counts

def getLabel(tuple):
    return tuple[-1]


value_label_counts = getValueLabelCounts(training_set, 'XB')
print(calcEntropy(value_label_counts[0]))

# print(training_set)


