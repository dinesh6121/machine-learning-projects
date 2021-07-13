import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()
test_idx = [0,50,100]

#training data... remove test_idx rows from training later use them to test if classifier is working
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#training the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

#exporting decision tree to a pdf
import pydotplus
from io import StringIO


dotfile = StringIO()
tree.export_graphviz(
    clf,
    out_file=dotfile,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    impurity=False
)

graph = pydotplus.graph_from_dot_data(dotfile.getvalue())   
graph.write_pdf('tree.pdf')