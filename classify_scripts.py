import os
import re

# Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import numpy as np

def get_texts(remove_stop=False):
	sws = [folder for folder in os.listdir('data') if not folder.startswith('.') ]

	sw_scripts = {}
	
	for sw in sws:
		base = 'data/'+sw
		# Read in the raw files
		scripts = [open(base+"/"+file, 'r').read() for file in os.listdir(base) if not file.startswith('.')]
		# Remove character names, settings, etc that are listed in all caps
		scripts = [re.sub(r'\b[A-Z]+\b', '', script) for script in scripts]
		# Remove line breaks
		scripts = [re.sub('\n', ' ', script) for script in scripts]
		# Remove excess spaces
		scripts = [re.sub(' +', ' ', script) for script in scripts]
		# Assign the script to the 
		sw_scripts[sw.split('.')[0]] = scripts
	
	return sw_scripts


# Change the chunk length to get a different sized word byte for training
def get_data(chunk_length=80):
	sws = get_texts()
	
	data = []
	target = []

	n = 0
	for sw in sws:
		for script in sws[sw]:
			cur = script.split(' ')
			# break the text into 80-word chunks
			chunks = [' '.join(cur[x:x+chunk_length]) for x in range(0, len(cur), chunk_length)]
			data += chunks
			target += [n]*len(chunks)
		n+=1

	return data, target

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

	results = gs.cv_results_

	predicted_test = gs.predict(test_data)
	avg_test = np.mean(predicted_test == test_target)

	predicted_train = gs.predict(train_data)
	avg_train = np.mean(predicted_train == train_target)

	return avg_test, avg_train, results


def predict_scripts():
	X, y = get_data()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

	avg_test, avg_train, results = test_model(LogisticRegression(), X_train, y_train, X_test, y_test)

	print(avg_test)
	print(avg_train)


predict_scripts()

