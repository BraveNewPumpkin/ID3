import pandas as pd
import sys
from pathlib import Path
from treelib import Node, Tree
from math import log2

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
print(header)

num_tuples = len(training_set[header[-1]])
num_class = {}


for val in training_set[header[-1]]:
    if val in num_class:
        num_class[val] += 1
    else:
        num_class[val] = 1

probability_class = {}
for key in num_class:
    probability_class[key] = num_class[key] / num_tuples

previous_entropy = 0
for key in probability_class:
    previous_entropy += -1 * probability_class[key] * log2(probability_class[key])
print(previous_entropy)

# print(training_set)



tree = Tree()

tree.create_node('Root', 'root')
tree.create_node('A', 'a', parent='root')
tree.create_node('B', 'b', parent='root')
tree.create_node('C', 'c', parent='root')

# tree.show(line_type="ascii-em")