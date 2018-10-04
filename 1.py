import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

dat = pd.read_csv('1.csv')


def custom_round(x, base = 50):
    return int (base * round(float(x)/base))

header = ['Id', #1
          'Sex', #2
          'Race', #3
          'First Gen', #4 0:no, 1:yes
          'SATReading', #5
          'SATMath', #6
          'High School GPA', #7
          'Semester Taken',#8
          'GPA',#9
          'Major Type',#10
          'Grade Intro Course',#11
          'Grade Follow Up Course',#12
          'Grade Fundamentals Course',#13
          'Grade Systems Course',#14
          'Grade Software Course',#15
          'Grade Paradigm Course',#16
          'Instructor Intro',#17
          'Instructor Follow Up',#18
          'Instructor Fundamentals',#19
          'Instructor Systems'#20
        ]

dat.columns = header

#select the input variables
my_features_names = ['SATReading','SATMath','High School GPA', 'GPA']
#set the dataframe according to the input variables
df = pd.DataFrame(dat, columns = my_features_names)

#clear the NaN data
df = df.dropna(axis=0)

x1 = df['SATReading'].apply((lambda x: custom_round(x, base=50))).values
x2 = df['SATMath'].apply((lambda x: custom_round(x, base=50))).values
x3 = df['High School GPA'].round(1).values
#set X as the data
X = pd.DataFrame(x1, columns=['SATReading'])
X['SATMath'] = x2
X['High School GPA'] = x3

#set y as target data
y = np.where(df['GPA'] >= 3.25, 1, 0)

#split data into 70:30 train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y), len(y))
print('Labels counts in y_train:', np.bincount(y_train), len(y_train))
print('Labels counts in y_test:', np.bincount(y_test),len(y_test))
print('Labels counts in X:', len(X))
print('Labels counts in X_train:', len(X_train))
print('Labels counts in X_test:', len(X_test))

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2) 
forest.fit(X_train, y_train)

print("Importances: ", list(zip(X_train, forest.feature_importances_)))

#see accuracy
from sklearn.metrics import accuracy_score

#predict out of the training data set
test_preds = forest.predict(X_test)
train_preds = forest.predict(X_train)

#get the actual goal attributes of the testing data set
test_true = y_test
train_true = y_train
print("Test Data Accuracy: ", accuracy_score(test_true, test_preds))
print("Train Data Accuracy: ", accuracy_score(train_true, train_preds))

##################################################
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=3, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Bad GPA',
                                        'Good GPA'],
                           feature_names=['SAT Reading',
                                        'SAT Math',
                                        'High School GPA'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('1_2.png') 