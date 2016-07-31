import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
import numpy as np

#incarcam datele predefinite in scikit-learn
iris = datasets.load_iris()
classifier = skflow.DNNClassifier(hidden_units=[10,20,10],n_classes=3)
classifier.fit(iris.data, iris.target,steps=200, batch_size=32)

score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print ("Acuratete: %f"%score)

# clasificam 2 flori
new_floare = np.array([[6.4,3.2,4.5,1.5], [5.8,3.1,5.0,1.7]], dtype=float)

y = classifier.predict(new_floare)
for i in range(len(y)):
	if y[i] == 0:
		print('setosa')
	elif y[i] == 1:
		print('versicolor')
	elif y[i] == 2:
		print('virginica')
