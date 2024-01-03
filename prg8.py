from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix 
iris = load_iris() 
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.1) 
print("Dataset is split into training and testing .. ") 
print("Size of training data and it's label",x_train.shape,y_train.shape) 
print("Size of test data and it's label",x_test.shape,y_test.shape) 
for i in range(len(iris.target_names)): 
    print("Labell", i, "-" ,str(iris.target_names[i])) 
classifier = KNeighborsClassifier(n_neighbors = 1) 
classifier.fit(x_train,y_train) 
y_pred = classifier.predict(x_test) 
print("Results of classification usin knn with k=1 ") 
for r in range(len(x_test)): 
    print("Sample :" ,str(x_test[r]), "Actual - Label" ,str(y_test[r]), "Predicted - Label" ,str(y_pred[r]))
print("Classifictaion Accuracy :" ,classifier.score(x_test,y_test)) 
print("Accuracy Metrices :") 
print(classification_report(y_test,y_pred)) 
print("Confusion Matrix:") 
print(confusion_matrix(y_test,y_pred))