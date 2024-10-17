import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
df=pd.read_csv("diabetes.csv")
print(df.head())
x=df.drop("Outcome",axis=1).values
y=df['Outcome'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model = AdaBoostClassifier(n_estimators=20,random_state=10)
model.fit(x_train,y_train)
train_score=model.score(x_train,y_train)
print(f"Training Score:{train_score}")
y_pred=model.predict(x_test)
test_accuracy=accuracy_score(y_test,y_pred)
print(f"Test accuracy:{test_accuracy}")
precision_score = precision_score(y_test,y_pred)
print("precision score:",precision_score)
recall_score = recall_score(y_test,y_pred)
print("recall_score:",recall_score)
f1=f1_score(y_test,y_pred)
print(f" F1 score:{f1}")
print("\n Classification report")
print(classification_report(y_test,y_pred))