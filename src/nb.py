from __future__ import print_function, division
import numpy as np
import pandas as pd
# import re

from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
import os, sys, re
from operator import add, truediv
import numpy as np
from functools import reduce

# User define functions
def remove_labels(x):
    '''
    Remove redundant labels we don't need. That is, we only keep labels ending with "CAT".
    '''
    text, labels = x
    new_labels = [i for i in labels.split(',') if i[-3:] == "CAT"]
    return [text, new_labels]

def word_count_cat(cat_name, rdd):
    '''
    Return word count per category among all text. cat_name is the name of the category (type:str). rdd is the collection of all text (type: RDD).
    Return a rdd with all lower case letter and remove the quotation sign (we can remove more stopwords later. Just change the 'quot' into sw list).
    '''
    return rdd.filter(lambda x: x[1] == cat_name).map(lambda x: x[0]).\
    map(lambda x: re.findall('\w+', x)).map(lambda x: [i.lower() for i in x if i.isalpha() and i not in swlist.value and len(i) > 1])

def word_count_all(text):
    '''
    Return word counts of all words
    '''
    return text.map(lambda x: x[0]).map(lambda x: re.findall('\w+', x)).\
    map(lambda x: [i.lower() for i in x if i.isalpha() and i not in swlist.value and len(i) > 1]).flatMap(lambda x: ([v, 1] for v in x)).reduceByKey(add)

def word_count_test(text):
    '''
    Return word counts of all words
    '''
    return text.map(lambda x: re.findall('\w+', x)).\
    map(lambda x: [i.lower() for i in x if i.isalpha() and i not in swlist.value and len(i) > 1])#.flatMap(lambda x: ([v, 1] for v in x)).reduceByKey(add)

def add_missing(cat, all_dict):
    '''
    Not every document (or every type of document) has every word in the vocab. So we need to add the word that are missing and put 0 count for them.
    '''
    cat = cat.flatMap(lambda x: ([v, 1] for v in x)).reduceByKey(add)#.mapValues(list)#.map(lambda x: x[1]).reduceByKey(add)
    missing = all_dict.subtractByKey(cat).map(lambda x: (x[0], 0)) #add 0 counts into dataset
    cat = cat.union(missing).map(lambda x: (x[0], x[1])) # add one to avoid 0
    return cat

def get_prob(x):
    '''
    Get conditional probabilities for each word in a category. Add one to avoid 0.
    '''
    x_count = x.count()
    x = add_missing(x, all_count).map(lambda x: (x[0], (x[1] + 1) / (x_count + TOTAL_WORD.value)))
    return x


if __name__ == "__main__":

    # Build a Spark Session
    spark = SparkSession\
        .builder\
        .appName("nb")\
        .getOrCreate()


    # Read in the training files
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    text_path = os.path.join(script_dir, 'X_train_large.txt')
    label_path = os.path.join(script_dir, 'y_train_large.txt')

    sw_path = os.path.join(script_dir, 'stopwords.txt')
    sw = spark.sparkContext.textFile(sw_path)
    swlist = spark.sparkContext.broadcast(sw.collect())

    label = spark.sparkContext.textFile(label_path)
    all_text = spark.sparkContext.textFile(text_path)

    #preprocessing the training set
    label = label.zipWithIndex().map(lambda x: [x[1], x[0]])   #Use zipWithIndex to get the indices and swap value and key
    all_text = all_text.zipWithIndex().map(lambda x: [x[1], x[0]])     #Use zipWithIndex to get the indices and swap value and key
    all_text = all_text.join(label).sortByKey().map(lambda x: x[1])    #Join texts and labels based on the key; sort the list, and delete the key column
    all_text = all_text.map(remove_labels).flatMap(lambda x: ((x[0], i) for i in x[1]))   #Remove redundant labels and flatten texts with multiple labels

    #Count the numbers of total texts and calculate the prior probability of each
    all_label = all_text.map(lambda x: x[1])
    LABEL_COUNT = spark.sparkContext.broadcast(len(all_label.collect()))
    all_label = all_label.map(lambda x: (x, 1)).reduceByKey(add).map(lambda x: (x[0], x[1]/LABEL_COUNT.value))
    prior_prob = all_label.collectAsMap()  #return a dict of label and its prior probability

    prior_values = np.array([prior_prob['CCAT'], prior_prob['ECAT'], prior_prob['GCAT'], prior_prob['MCAT']], dtype='float128')  #make sure the order of the dict is correct.


    all_count = word_count_all(all_text) # Count all the words
    TOTAL_WORD = spark.sparkContext.broadcast(all_count.map(lambda x: x[0]).count()) #broadcast the total word count value

    ccat = word_count_cat('CCAT', all_text)
    ecat = word_count_cat('ECAT', all_text)
    gcat = word_count_cat('GCAT', all_text)
    mcat = word_count_cat('MCAT', all_text)


    # get the probabilities for each category
    ccat = get_prob(ccat)
    ecat = get_prob(ecat)
    gcat = get_prob(gcat)
    mcat = get_prob(mcat)

    total_prob = ccat.union(ecat).union(gcat).union(mcat).groupByKey().mapValues(list)  #union all categories together

    #put the result into a dictionary. Key is each word and the value is 4-place tuple (which gives the conditional probability of a word given a category).
    total_prob = total_prob.collectAsMap()

    # Read the test file and get the results
    test_path = os.path.join(script_dir, 'X_test_large.txt')
    test_text = spark.sparkContext.textFile(test_path)
    test_text = word_count_test(test_text)
    test_prob = test_text.map(lambda x: [total_prob[i] for i in x if i in total_prob])


    test_prob = np.array(test_prob.collect())
    test_prob = np.array([reduce(lambda x, y: x * y, np.array(i, dtype = 'float128')) for i in test_prob]) # calculate the prob for each doc given words in each doc
    test_prob = np.array([i * prior_values for i in test_prob]) #multiply value of each doc with the prior prob of each category
    maxidx = np.argmax(test_prob, axis = 1) # find the maximum idx number
    cat_dict = {0: 'CCAT', 1: 'ECAT', 2: 'GCAT', 3: 'MCAT'} # hand-code dict type. The order is based on the order of union operations earlier
    res = [cat_dict[i] for i in maxidx]

    # print (res)


    # Testing
    test_res_path = os.path.join(script_dir, 'y_test_large.txt')

    with open(test_res_path, 'w') as file:      
        file.write("\n".join(res).lower())

    # label = label.splitlines()
    # accu = 0
    # for i, v in enumerate(res):
    #     if v in label[i]:
    #         accu += 1
    # print (accu/len(res))


    ##Todo -- change all num into "NUM"
