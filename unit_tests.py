# This is the unittest file for all our methods in pre-processing, training, and testing. 

import pyspark
import unittest
from testing import *
from pre-processing import *

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


class PreprocessingMethods(unittest.TestCase):
    
    def test_RemoveEcxeptAlphabets(self):
        self.assertEqual('a1b,c.d!e?g$h'.upper(), 'ABCDEFGH')
    
    def test_MinimumLength(self):
        new_line = MinimumLength('this is a test to remove word with length less than two', 3)
        self.assertEqual(new_line, 'this test remove word with length less than two')
    
    def test_NotStopWords(self):
        stopwords_List = ['group', 'I', 'like']
        result = NotStopWords('I like working in my data science group, it is cool.', stopwords_List)
        self.assertEqual(result, 'working in my data science, it is cool.')

    def test_X_Preprocessing(self):
        sc = SparkContext.getOrCreate()
        rdd = sc.parallelize(["A dedicated &quot;snow desk&quot; has been set up by the New York and New Jersey Port"])
        new_rdd = X_Preprocessing(rdd , 2)
        result =  [['dedicated','set','york','jersey','port']]
        self.assertEqual(new_rdd.collect(), result_rdd)

    def test_y_Preprocessing(self):
        sc = SparkContext.getOrCreate()
        rdd = sc.parallelize['C11,C24,CCAT,GCAT,GWEA']
        new_rdd = X_Preprocessing(rdd)
        result =  [['CCAT', 'GCAT']]
        self.assertEqual(new_rdd.collect(), result_rdd)
        

if __name__ == "__main__":
	sc = pyspark.SparkContext()
	unittest.main()