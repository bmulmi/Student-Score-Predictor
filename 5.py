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

dat = pd.read_csv('5.csv')

header = ['Id', #1 A
        'Sex', #2 B
          'Race', #3 C 
          'First Gen', #4 D 
          'SAT Reading', #5 E
          'SAT Math', #6 F
          'High School GPA', #7 G
          'Semester Taken',#8 H 
          'GPA',#9 I
          'Major Type',#10 J
          'Grade Intro Course',#11 K
          'Grade Follow Up Course',#12 L
          'Grade Fundamentals Course',#13 M
          'Grade Systems Course',#14 N 
          'Grade Software Course',#15 O
          'Grade Paradigm Course',#16 P
          'Instructor Intro',#17 Q
          'Instructor Follow Up',#18 R
          'Instructor Fundamentals',#19 S
          'Instructor Systems'#20 T
         ]

dat.columns = header

#select the input variables
my_features_names = ['Grade Fundamentals Course', 'Grade Systems Course', 'Grade Software Course', 'Grade Paradigm Course']

#set the dataframe according to the input variables
df = pd.DataFrame(dat, columns = my_features_names)

#clear the NaN data
df = df.dropna(axis=0)

#set X as the data
X = pd.DataFrame(df, columns = ['Grade Fundamentals Course', 'Grade Systems Course', 'Grade Software Course']).values

#set y as target data
y = np.where(df['Grade Paradigm Course'] >= 3.0, 1, 0)
y

#split data into 70:30 train:test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in (goal) attributes y:', np.bincount(y), len(y))
print('Labels counts in (goal) y_train:', np.bincount(y_train), len(y_train))
print('Labels counts in (goal) y_test:', np.bincount(y_test),len(y_test))
print('Labels counts in (features) X:', len(X))
print('Labels counts in (features) X_train:', len(X_train))
print('Labels counts in (features) X_test:', len(X_test))

##############################################
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

##############################################
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=5, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Bad Grades',
                                        'Good Grades'],
                           feature_names=['Fundamentals',
                                        'Systems',
                                        'Software'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('5.png') 