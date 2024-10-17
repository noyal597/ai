import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
df=pd.read_csv("diabetes.csv")
print(df.head())
x=df.drop("Outcome",axis=1).values
y=df["Outcome"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=0)
kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(x_train,y_train)
y_pred=kn.predict(x_test)
test = accuracy_score(y_test,y_pred)
print("test accuracy",test)
precision = precision_score(y_test,y_pred)
print("Precision:",precision)
recall = recall_score(y_test,y_pred)
print("recall score:",recall)
f1=f1_score(y_test,y_pred)
print("F1 score:",f1)
print("Classification report \n")
print(classification_report(y_test,y_pred))