from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.naive_bayes import GaussianNB
iris = load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred,average="macro"))
print("Recall Score:",recall_score(y_test,y_pred,average="macro"))
print("f1 Score:",f1_score(y_test,y_pred,average="macro"))
print("Classification Report",classification_report(y_test,y_pred))