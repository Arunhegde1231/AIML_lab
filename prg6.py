from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split as s

dataset=load_iris()
x=dataset.data
y=dataset.target
x_train,x_test, y_train, y_test= s(x,y,test_size=0.2, random_state=1)
gnb=GaussianNB()
classifier=gnb.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print("accuracy matrices : \n" ,metrics.classification_report(y_test,y_pred))
print("accuracy of matrices is : ", metrics.accuracy_score(y_test,y_pred))
print("confusion matrix ")
print("accuracy matrices : ", metrics.confusion_matrix(y_test,y_pred))