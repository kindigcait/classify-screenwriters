import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def clean_text(text):
	

	text = re.sub(r'\b[A-Z]+\b', '', text)
	# Remove line breaks
	text = re.sub('\n', ' ', text)
	# Remove excess spaces
	text = re.sub(' +', ' ', text)

	text = set(text.split())

	text = list(text - stopwords)

	return text

chunk_length = 80
genre_tags = ['genre1', 'genre2', 'genre3', 'genre4']
screenwriters = [folder for folder in os.listdir('data') if not folder.startswith('.') ]

meta = []
for screenwriter in screenwriters:
	base = 'data/'+screenwriter
	# Get the titles
	titles = [file for file in os.listdir(base) if not file.startswith('.')]


	for title in titles:
		f = open(base+"/"+title, 'r')
		genres = f.readline().strip()
		director = f.readline().strip()
		production_company = f.readline().strip()
		genres = genres.split(',')

		cur = {}
		cur['title'] = title.split('.')[0]
		cur['screenwriter'] = screenwriter
		cur['director'] = director
		cur['production_company'] = production_company

		for i in range(0, len(genre_tags)):
			try:
				cur[genre_tags[i]] = genres[i]
			except:
				cur[genre_tags[i]] = None

		script = clean_text(f.read())

		for i in range(0, len(script), chunk_length):
			temp = dict(cur)
			text = ' '.join(script[i:i+chunk_length])
			temp['text'] = text
			meta.append(temp)

df = pd.DataFrame(meta)
print(df)
df.to_csv('metadata_split.csv')


