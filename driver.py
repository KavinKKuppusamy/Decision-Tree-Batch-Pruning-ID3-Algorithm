from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import random,copy

print(" Give link of csv file or path of the uci dataset without quotes and spaces")
# https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data 
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
csv_myinput = input()
#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

#header = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square','middle-right-square','bottom-left-square','bottom-middle-square','bottom-right-square','Class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None, names=['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square','middle-right-square','bottom-left-square','bottom-middle-square','bottom-right-square','Class'])

df = pd.read_csv(csv_myinput)
header = list(df.columns.values)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
 
trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******\n")
#print_tree(t)
acc = computeAccuracy(test, t)
print("\n\nAccuracy on test before pruning = " + str(acc))



## TODO: You have to decide on a pruning strategy
best_tree = {"tree": copy.deepcopy(t), "accuracy": acc}
nodeid_list =[]
t_best = t
#Setting the pruning nodes limit as using n
n = 2
count = 0
#"Pruning_limit" limits the number of iterations in pruning the nodes
pruning_limit = 10
for node in innerNodes:
    if node.id != 0:
        nodeid_list.append(node.id)
random.shuffle(nodeid_list)
#with the shuffled inner nodes list , partitioning it into multiple list with length 'n'
print(len(nodeid_list))
partition_list = [nodeid_list[i * n:(i + 1) * n] for i in range((len(nodeid_list) + n - 1) // n)]
if len(partition_list) > 10:
    pruning_limit = len(partition_list)//2

for node_list in partition_list:
    if len(node_list) >= 1 and count <= pruning_limit:
        to_prune_tree = copy.deepcopy(t)
        t_pruned = prune_tree(to_prune_tree, node_list)
        after_pruned_accuracy = computeAccuracy(test, to_prune_tree)
        count += 1
        if after_pruned_accuracy > acc:
            best_tree["tree"] = copy.deepcopy(t_pruned)
            best_tree["accuracy"] = after_pruned_accuracy

print("\n\nThe Final Pruned Tree\n")
print(print_tree(best_tree["tree"]))
print("\nBest accuracy obtained is " + str(best_tree["accuracy"]))











