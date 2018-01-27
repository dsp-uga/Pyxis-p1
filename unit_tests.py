# This is the unittest file for all our methods in pre-processing, training, and testing.
import pyspark
import unittest
from pre_processing import *
from training import *
from testing import *
import math
import numpy as np

class TestTestingMethods(unittest.TestCase):

	def test_words_to_probs(self):

		documents = sc.parallelize((
						('doc1',('word1','word2','word3')),
						('doc2',('word2','word4')),
						('doc3',('word3'))
					))

		word_probabilities_dictionary = {
						'word1': (1,0,0,0),
						'word2': (0,1,1,0),
						'word3': (0,0,1,1),
						'word4': (0,0,0,1)
					}

		expected_result = sc.parallelize((
						('doc1',[(1,0,0,0),(0,1,1,0),(0,0,1,1)]),
						('doc2',[(0,1,1,0),(0,0,0,1)]),
						('doc3',(0,0,1,1))
					))

		self.assertEqual(
				words_to_probs(documents,word_probabilities_dictionary).collect(),
				expected_result.collect()
			)

	def test_docs_to_probs(self):
		documents = sc.parallelize((
						('doc1',[(1,0,0,0),(0,1,1,0),(0,0,1,1)]),
						('doc2',[(0,1,1,0),(0,0,0,1)]),
						('doc3',(0,0,1,1))
					))

		priors_dictionary = {'CCAT': 1, 'ECAT': 2, 'GCAT': 3, 'MCAT': 4}

		expected_result = sc.parallelize((
						('doc1',[2,3,5,5]),
						('doc2',[1,3,4,5]),
						('doc3',[1,2,4,5])
					))

		self.assertEqual(
				docs_to_probs(documents,priors_dictionary).collect(),
				expected_result.collect()
			)

	def test_class_preds(self):
		documents = sc.parallelize((
						('doc1',[2,3,5,5]),
						('doc2',[1,7,4,5]),
						('doc3',[6,2,4,5]),
						('doc4',[0,0,0,1])
					))

		expected_result = sc.parallelize((
						('doc1',2),
						('doc2',1),
						('doc3',0),
						('doc4',3)
					))

		self.assertEqual(class_preds(documents).collect(), expected_result.collect())

	def test_accuracy(self):
		predictions = sc.parallelize((
						('doc1',2),
						('doc2',1),
						('doc3',0),
						('doc4',3)
					))

		labels = sc.parallelize((
						('doc1','GCAT'),
						('doc2','ECAT'),
						('doc3','CCAT'),
						('doc4','MCAT')
					))

		expected_result = 1.0

		self.assertEqual(accuracy(predictions,labels), expected_result)


# class PreprocessingMethods(unittest.TestCase):
#
#     def test_RemoveEcxeptAlphabets(self):
#         self.assertEqual('a1b,c.d!e?g$h'.upper(), 'ABCDEFGH')
#
#     def test_MinimumLength(self):
#         new_line = MinimumLength('this is a test to remove word with length less than two', 3)
#         self.assertEqual(new_line, 'this test remove word with length less than two')
#
#     def test_NotStopWords(self):
#         stopwords_List = ['group', 'I', 'like']
#         result = NotStopWords('I like working in my data science group, it is cool.', stopwords_List)
#         self.assertEqual(result, 'working in my data science, it is cool.')
#
#     def test_X_Preprocessing(self):
#         sc = SparkContext.getOrCreate()
#         rdd = sc.parallelize(["A dedicated &quot;snow desk&quot; has been set up by the New York and New Jersey Port"])
#         new_rdd = X_Preprocessing(rdd , 2)
#         result =  [['dedicated','set','york','jersey','port']]
#         self.assertEqual(new_rdd.collect(), result)
#
#     def test_y_Preprocessing(self):
#         sc = SparkContext.getOrCreate()
#         rdd = sc.parallelize(['C11,C24,CCAT,GCAT,GWEA'])
#         new_rdd = X_Preprocessing(rdd)
#         result =  [['CCAT', 'GCAT']]
#         self.assertEqual(new_rdd.collect(), result)


class TestTraining(unittest.TestCase):

	def test_words_to_probs(self):
		rdd_label = sc.parallelize([['CCAT', 'MCAT'], ['ECAT'], ['GCAT']])
		rdd_text = sc.parallelize([['word1', 'word2'], ['word3'], ['word5']])
		returned_label = ['CCAT', 'MCAT', 'ECAT', 'GCAT']
		returned_text = [(['word1', 'word2'], 'CCAT'), (['word1', 'word2'], 'MCAT'), (['word3'], 'ECAT'), (['word5'], 'GCAT')]
		self.assertEqual(process_label_text(rdd_label, rdd_text)[0].collect(), returned_label)
		self.assertEqual(process_label_text(rdd_label, rdd_text)[1].collect(),returned_text)   # should just have use assertTupleEqual instead...

	def test_add_missing(self):
		cat = sc.parallelize([['word1'], ['word1', 'word2'], ['word3']])
		all_dict = sc.parallelize([['word1', 3], ['word4', 5], ['word2', 2], ['word3', 1]])
		total_cat = [('word1', 2), ('word2', 1), ('word3', 1), ('word4', 0)]
		self.assertEqual(add_missing(cat, all_dict).collect(), total_cat)

	def test_count_label(self):
		all_cat = sc.parallelize(['CCAT', 'MCAT', 'CCAT', 'MCAT', 'MCAT'])
		all_prior = {'CCAT': math.log(0.4), 'MCAT': math.log(0.6)}
		self.assertDictEqual(count_label(all_cat, 5).collectAsMap(), all_prior)

	def test_get_prob(self):
		total_vocab = sc.parallelize([['word1', 2], ['word2', 3], ['word3', 1]])
		cat_vocab = sc.parallelize([['word1', 'word2'], ['word2']])
		cat_return = [('word1', 2/9), ('word2', 3/9), ('word3', 1/9)]
		self.assertEqual(get_prob(cat_vocab, 3, 6, total_vocab).collect(), cat_return)

	def test_get_total_vocab(self):
		training_text = sc.parallelize([['word1', 2], ['word2', 3]])
		testing_text = sc.parallelize([['word1', 'word1', 'word3'], ['word4']])
		total_v = [('word1', 4), ('word2', 3), ('word3', 1), ('word4', 1)]
		self.assertEqual(get_total_vocab(training_text, testing_text).sortByKey().collect(), total_v) #Need to sort by key here.....

	def test_word_count_cat(self):
		rdd = sc.parallelize([[['foo'], 'CCAT'], [['bar'], 'GCAT'], [['word', 'word'], 'CCAT']])
		cat_text = [['foo'], ['word', 'word']]
		self.assertEqual(word_count_cat('CCAT', rdd).collect(), cat_text)

if __name__ == "__main__":
	sc = pyspark.SparkContext()
	unittest.main()
