'''
assuming documents takes the form:
(
('word1', 'word2', ... , 'wordN'),
('word1', 'word2', ... , 'wordN'),
... ,
('word1', 'word2', ... , 'wordN')
)
'''

def unique_words(document):
	uniques = {}
	for word in document:
		if word not in uniques:
			uniques[word] = 1
	return [(k,v) for k, v in uniques.items()]

def term_frequency(document):
	uniques = {}
	for word in document:
		if word in uniques:
			uniques[word] += (float)(1/len(document))
		else:
			uniques[word] = (float)(1/len(document))
	return [(k,v) for k, v in uniques.items()]

def TF(documents):
	return documents.map(term_frequency)

def IDF(documents):
	num_docs = documents.count()
	word_docs = documents.flatMap(unique_words).reduceByKey(lambda x, y: x+y)
	IDF = word_docs.map(lambda x: (x[0], math.log(num_docs/x[1])))
	return IDF

def TF_IDF(documents):
	tf = TF(documents)
	idf = IDF(documents).collectAsMap()
	tf_idf = tf.map(lambda x: [(y[0], y[1]*idf[y[0]]) for y in x])
	return tf_idf