import tensorflow.contrib.learn as skflow
import tensorflow as tf
import numpy as np

IRIS_TRAINIG = "data/iris_training.csv"
IRIS_TEST = "data/iris_test.csv"

training_set = skflow.datasets.base.load_csv(filename=IRIS_TRAINIG,target_dtype=np.int)
test_set = skflow.datasets.base.load_csv(filename=IRIS_TEST,target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, training_set.target, test_set.target

classifier = skflow.DNNClassifier(hidden_units=[10,20,10], n_classes=3)

classifier.fit(x=x_train, y=y_train, steps=200, batch_size=32)

accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print("Accuracy {0:f}".format(accuracy_score))

new_flower = np.array([[6.4,3.2,4.5,1.5], [5.8,3.1,5.0,1.7]], dtype=float)

y = classifier.predict(new_flower)

for i in range(len(y)):
	if y[i] == 0:
		print('setosa')
	elif y[i] == 1:
		print('versicolor')
	elif y[i] == 2:
		print('virginica')