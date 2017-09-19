import pandas as pd
from pathlib import Path
from treelib import Node, Tree

input_dirpath = Path('D:','Users','Kyle','Downloads','school','machine_learning','data_sets1')

training_set = pd.read_csv(Path(input_dirpath, 'training_set.csv'))

print(training_set)



tree = Tree()

tree.create_node('Root', 'root')
tree.create_node('A', 'a', parent='root')
tree.create_node('B', 'b', parent='root')
tree.create_node('C', 'c', parent='root')

tree.show(line_type="ascii-em")