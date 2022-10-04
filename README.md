# Implementation-of-k-NN-algorithm-in-python-code-for-class-knn-and-its-methods

**kNN algorithm** is used for solving classification model problems (Supervised Learning).
Here, the python code consists of basic implementation of knn algorithm similar to **'KNeighborsClassifier'** function in **'sklearn.neighbors'** module.
This is just a gist of how the code on the inside of 'KNeighborsClassifier' might look like and give basic idea about how it works.

**'kNN_mymodule.py'** module consists of class KNN and it has some functions defined:
_fit_ : similar to 'KNeighborsClassifier'
_find_distance_ : returns distance matrix (dist between inputs and train data), 
_k_neighbors_ : first k nearest neighbors corresponding to input data , 
_predict_ : similar to predict in 'KNeighborsClassifier' , 
_evaluate_ : similar to score in 'KNeighborsClassifier' .

**.ipynb** file gives walkthrough of how to use the module by giving an example and comparing it with classical function 'KNeighborsClassifier'.


