import TreeMaker as tm
import pandas as pd
import sys
import pprint as pp
from pathlib import Path
from math import log2

usage = 'Main.py /path/to/training/dataset /path/to/validation/dataset /path/to/testing/dataset pruning_factor'

if len(sys.argv) < 5:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 5:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)

def main(argv):
    training_data_path = Path(sys.argv[1])
    validation_data_path = Path(sys.argv[2])
    test_data_path = Path(sys.argv[3])
    pruning_factor = sys.argv[4]

    training_set = pd.read_csv(training_data_path)
    validation_set = pd.read_csv(validation_data_path)
    test_set = pd.read_csv(test_data_path)

    headers = list(training_set.columns.values)

    print('Number of training attributes = %d' % len(headers))
    starting_entropy = calcStartingEntropy(training_set, headers)

    tree = tm.makeTree(training_set, headers, starting_entropy)

    print('Total number of nodes in the tree = %d' % tree.size())
    print('Number of leaf nodes in the tree = %d' % len(tree.leaves()))
    print('Max depth of tree = %d' % tree.depth())

    training_set_accuracy = calcAccuracyForData(training_set, headers, tree)

    print('Accuracy of the model on the training dataset = %.2f%%' % training_set_accuracy)

    return 0

def calcStartingEntropy(set, dims):
    starting_entropy = 0
    num_rows = len(set[dims[-1]])
    print('Number of training instances = %d' % num_rows)
    num_label = {}

    #find initial entropy
    for val in set[dims[-1]]:
        if val in num_label:
            num_label[val] += 1
        else:
            num_label[val] = 1

    probability_label = {}
    for key in num_label:
        probability_label[key] = num_label[key] / num_rows

    for key in probability_label:
        starting_entropy += -1 * probability_label[key] * log2(probability_label[key])
    print('starting entropy of data: %f' % starting_entropy)
    return starting_entropy

def calcAccuracyForData(set, dims, tree):
    root_node = tree.get_node(tree.root)
    total_instances = set.shape[0]
    num_correct = calcNumCorrectlyClassifiedForData(set, dims, tree, root_node)
    accuracy = num_correct / total_instances
    return accuracy

def calcNumCorrectlyClassifiedForData(set, dims, tree, current_node):
    #TODO base cases: leaf node or no node exists for value


    dim = current_node.tag
    #traverse tree checking accuracy at leaf nodes
    dim_value_subsets = seperateSetByDimValues(set, dim)
    #TODO recurse on each child checking if child exists


def seperateSetByDimValues(set, dim):
    subsets = {}
    #get unique values for dim
    unique_values = getattr(set, dim).unique()
    for value in unique_values:
        mask = set[dim].values == value
        subsets[value] = set[mask]
    return subsets

main(sys.argv)



