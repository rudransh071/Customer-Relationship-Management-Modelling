from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

x_train, y_train, x_test, y_test = preprocessing.get_data()
model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
