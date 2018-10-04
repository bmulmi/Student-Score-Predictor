import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


def custom_round(x, base = 50):
    return int (base * round(float(x)/base))

dat = pd.read_csv('3.csv')

header = ['Id', #1 A
          'Sex', #2 B
          'Race', #3 C 
          'First Gen', #4 D 
          'SATReading', #5 E
          'SATMath', #6 F
          'High School GPA', #7 G
          'Semester Taken',#8 H 
          'GPA',#9 I
          'Major Type',#10 J 1=CS, 2=Math, 3=Science, 4=Non-Science
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
my_features_names = ['SATReading','SATMath','High School GPA', 'Grade Intro Course']

#set the dataframe according to the input variables
df = pd.DataFrame(dat, columns = my_features_names)

#clear the NaN data
df = df.dropna(axis=0)

#transform the data
x1 = df['SATReading'].apply((lambda x: custom_round(x, base=50))).values
x2 = df['SATMath'].apply((lambda x: custom_round(x, base=50))).values
x3 = df['High School GPA'].round(1).values

#set X data
X = pd.DataFrame(x1, columns=['SATReading'])
X['SATMath'] = x2
X['High School GPA'] = x3

#set y as target data
y = np.where(df['Grade Intro Course'] >= 3.25, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1, stratify=y)

print('Labels counts in (goal) attributes y:', np.bincount(y), len(y))
print('Labels counts in (goal) y_train:', np.bincount(y_train), len(y_train))
print('Labels counts in (goal) y_test:', np.bincount(y_test), len(y_test))

print('Labels counts in (features) X:', len(X))
print('Labels counts in (features) X_train:', len(X_train))
print('Labels counts in (features) X_test:', len(X_test))


#*****************************************************************************
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

#*****************************************************************************
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=3, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Bad',
                                        'Good'],
                           feature_names=['SAT Reading',
                                        'SAT Math',
                                        'H.S. GPA'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('3.png') 