# This is the unittest file for all our methods in pre-processing, training, and testing. 

import pyspark
import unittest
from testing import *

class TestStringMethods(unittest.TestCase):

	def test_words_to_probs(self):
		self.assertEqual(
				words_to_probs(
					sc.parallelize((
						('doc1',('word1','word2','word3')),
						('doc2',('word2','word4')),
						('doc3',('word3'))
					)),
					{
						'word1': (1,0,0,0), 
						'word2': (0,1,1,0), 
						'word3': (0,0,1,1), 
						'word4': (0,0,0,1)
					}
				).collect(),
				sc.parallelize((
						('doc1',[(1,0,0,0),(0,1,1,0),(0,0,1,1)]),
						('doc2',[(0,1,1,0),(0,0,0,1)]),
						('doc3',(0,0,1,1))
					)).collect()
			)




if __name__ == "__main__":
	sc = pyspark.SparkContext()
	unittest.main()