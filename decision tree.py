##decision tree##


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


col_names = ['Reservation', 'Raining', 'BadService', 'Saturday', 'Result']
hoteldata = pd.read_csv("hotels.csv", header=None, names=col_names)


feature_cols = ['Reservation', 'Raining', 'BadService', 'Saturday']
X = hoteldata[feature_cols]
y = hoteldata.Result


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1)


clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf.fit(X_train, Y_train)


y_pred = clf.predict(X_test)
print("ytest=", Y_test.values)  
print("ypred=", y_pred)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred))


plt.figure(figsize=(12, 8))  
plot_tree(clf, 
          filled=True, 
          rounded=True, 
          feature_names=feature_cols, 
          class_names=["Leave", "Wait"])
plt.title("Decision Tree Visualization")


plt.savefig('hotels_tree.png', bbox_inches='tight')  
plt.show()