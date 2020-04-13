# Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Models
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Graphing library
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Generalized tester function
# Usage: Pass a dictionary of (paramater, possible_tunings) key value pairs and GridSearch will optimize across them
def test_model(clf, train_data, train_target, test_data, test_target, tuned_parameters={}):
	text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', clf),
	])
	
	gs = GridSearchCV(text_clf, tuned_parameters, n_jobs = -1, cv=5, refit=True, return_train_score=True)
	
	gs.fit(train_data, train_target)

	return gs.predict(test_data), gs.classes_


def generate_confusion(labels, predictor, y_test, y_pred):
	conf_mat = confusion_matrix(y_test, y_pred)
	fig, ax = plt.subplots(figsize=(10,10))
	sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt='d',)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.yticks(rotation=0)
	plt.title('Actual and predicted '+predictor+' for screenplay snippets')

	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)

	plt.show()


# Predict the screenwriter of a script!
def predict_scripts():
	predictor = 'genre1'
	df = pd.read_csv('metadata_split_80.csv')

	df = df.dropna(subset=[predictor])
	
	X = df['text']
	y = df[predictor]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

	y_pred, classes = test_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

	print(accuracy_score(y_test, y_pred))
	generate_confusion(classes, predictor, y_test, y_pred)


predict_scripts()

