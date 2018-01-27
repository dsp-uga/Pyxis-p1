# This is the unittest file for all our methods in pre-processing, training, and testing. 

import pyspark
import unittest
from testing import *

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
		# todo

	def test_accuracy(self):
		# todo

if __name__ == "__main__":
	sc = pyspark.SparkContext()
	unittest.main()