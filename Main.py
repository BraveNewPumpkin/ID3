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

    starting_entropy = calcStartingEntropy(training_set, headers)

    tree = tm.makeTree(training_set, headers, starting_entropy)
    return 0

def calcStartingEntropy(set, dims):
    starting_entropy = 0
    num_rows = len(set[dims[-1]])
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


main(sys.argv)



