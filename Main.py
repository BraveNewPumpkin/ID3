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

training_data_path = Path(sys.argv[1])
validation_data_path = Path(sys.argv[2])
test_data_path = Path(sys.argv[3])
pruning_factor = sys.argv[4]

training_set = pd.read_csv(training_data_path)
validation_set = pd.read_csv(validation_data_path)
test_set = pd.read_csv(test_data_path)

headers = list(training_set.columns.values)



tree = tm.makeTree(training_set, headers)
