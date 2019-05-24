from sklearn import metrics
import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn import tree

x_train, y_train, x_test, y_test = preprocessing.get_data()

model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = np.asarray((y_pred))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
