'''
assuming documents takes the form:
(
('word1', 'word2', ... , 'wordN'),
('word1', 'word2', ... , 'wordN'),
... ,
('word1', 'word2', ... , 'wordN')
)
'''
import math

# removes duplicates and converts uniques to (word, 1) tuples
def unique_words(document):
	uniques = {}
	for word in document:
		if word not in uniques:
			uniques[word] = 1
	return [(k,v) for k, v in uniques.items()]

# replaces each word of a document with its (word, x) 
# where x is its proportional frequency in the document
def term_frequency(document):
	uniques = {}
	for word in document:
		if word in uniques:
			uniques[word] += (float)(1/len(document))
		else:
			uniques[word] = (float)(1/len(document))
	return [(k,v) for k, v in uniques.items()]

# gets the term frequency for each word in each document
def TF(documents):
	return documents.map(term_frequency)

# gets the inverse document frequency for each word in each document
def IDF(documents):
	num_docs = documents.count()
	word_docs = documents.flatMap(unique_words).reduceByKey(lambda x, y: x+y)
	IDF = word_docs.map(lambda x: (x[0], math.log(num_docs/x[1])))
	return IDF

# given a list of documents, this converts each word in each document to a tuple
# of the form (word, x), where word is the original word, and x is its tf-idf score
def TF_IDF(documents):
	tf = TF(documents)
	idf = IDF(documents).collectAsMap()
	tf_idf = tf.map(lambda x: [(y[0], y[1]*idf[y[0]]) for y in x])
	tf_idf = tf_idf.map(lambda x: [y for y in x if y[1]!=0.0]) #filter out 0 values
	return tf_idf
