import TreeMaker as tm
import pandas as pd
import sys
import random
from pprint import pprint
from pathlib import Path
from math import log2, trunc

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
    pruning_factor = float(sys.argv[4])

    training_set = pd.read_csv(training_data_path)
    validation_set = pd.read_csv(validation_data_path)
    test_set = pd.read_csv(test_data_path)

    headers = list(training_set.columns.values)

    print('Number of training attributes = %d' % len(headers))
    starting_entropy = calcStartingEntropy(training_set, headers)

    headers_copy = headers.copy()
    tree = tm.makeTree(training_set, headers_copy, starting_entropy)

    print('-'*80)
    print('Total number of nodes in the tree = %d' % tree.size())
    print('Number of leaf nodes in the tree = %d' % len(tree.leaves()))
    sum_of_leaf_depths = 0
    for leaf in tree.leaves():
        sum_of_leaf_depths += tree.depth(leaf)
    print('Average depth of tree = %f' % (sum_of_leaf_depths / len(tree.leaves())))
    print('Max depth of tree = %d' % tree.depth())

    training_set_accuracy = calcAccuracyForData(training_set, headers, tree)
    validation_set_accuracy = calcAccuracyForData(validation_set, headers, tree)
    test_set_accuracy = calcAccuracyForData(test_set, headers, tree)

    print('Accuracy of the model on the training dataset = %.2f%%' % training_set_accuracy)
    print('Accuracy of the model on the validation dataset = %.2f%%' % validation_set_accuracy)
    print('Accuracy of the model on the test dataset = %.2f%%' % test_set_accuracy)
    print('-'*80)
    headers_copy = headers.copy()
    random_tree = tm.makeRandomTree(training_set, headers_copy)

    print('Total number of nodes in the random_tree = %d' % random_tree.size())
    print('Number of leaf nodes in the random_tree = %d' % len(random_tree.leaves()))
    sum_of_leaf_depths = 0
    for leaf in tree.leaves():
        sum_of_leaf_depths += tree.depth(leaf)
    print('Average depth of tree = %f' % (sum_of_leaf_depths / len(tree.leaves())))
    print('Max depth of random_tree = %d' % random_tree.depth())

    training_set_accuracy = calcAccuracyForData(training_set, headers, random_tree)
    validation_set_accuracy = calcAccuracyForData(validation_set, headers, random_tree)
    test_set_accuracy = calcAccuracyForData(test_set, headers, random_tree)

    print('Accuracy of the random model on the training dataset = %.2f%%' % training_set_accuracy)
    print('Accuracy of the random model on the validation dataset = %.2f%%' % validation_set_accuracy)
    print('Accuracy of the random model on the test dataset = %.2f%%' % test_set_accuracy)

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
    #base cases: leaf node or no node exists for value
    if current_node is None:
        #NOTE: this is arbitrary behavior. Specification does not define behavior for nodes that don't have trained instances
        #randomly choose a class to be correct
        if set.shape[0] > 0:
            secure_random = random.SystemRandom()
            unique_classes = getattr(set, dims[-1]).unique()
            chosen_class = secure_random.choice(unique_classes)
            mask = set[dims[-1]] == chosen_class
            num_matching = set[mask].shape[0]
            return num_matching
        else:
            return 0
    elif current_node.is_leaf():
        mask = set[dims[-1]] == current_node.data.majority_class
        return set[mask].shape[0]

    dim = current_node.tag
    dim_value_subsets = seperateSetByDimValues(set, dim)
    sum_correct = 0
    current_node_children = tree.children(current_node.identifier)
    #recurse on each child checking if child exists
    for value, subset in dim_value_subsets.items():
        value_node_list = list(filter(lambda node: node.data.decision_value == value, current_node_children))
        value_node = value_node_list[0] if len(value_node_list) > 0 else None
        sum_correct += calcNumCorrectlyClassifiedForData(subset, dims, tree, value_node)
    return sum_correct


def seperateSetByDimValues(set, dim):
    subsets = {}
    #get unique values for dim
    unique_values = getattr(set, dim).unique()
    for value in unique_values:
        mask = set[dim].values == value
        subsets[value] = set[mask]
    return subsets

def pruneTree(tree, pruning_factor):
    num_nodes = tree.size()
    num_nodes_to_prune = trunc(num_nodes * pruning_factor)
    pruned_nodes = 0
    secure_random = random.SystemRandom()
    while pruned_nodes < num_nodes_to_prune:
        sufficiently_small_subtree_root_nodes = []
        for node in tree.all_nodes():
            subtree_size = tree.subtree(node.identifier).size()
            if subtree_size <= (num_nodes_to_prune - pruned_nodes):
                sufficiently_small_subtree_root_nodes.append(node)
        chosen_subtree_root_node = secure_random.choice(sufficiently_small_subtree_root_nodes)
        pruned_nodes += tree.remove_node(chosen_subtree_root_node.identifier)

    return tree

main(sys.argv)



