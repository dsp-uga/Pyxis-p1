from __future__ import print_function, division
import numpy as np
import pandas as pd
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
import os, sys, re, math
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
    cat = cat.flatMap(lambda x: ([v, 1] for v in x)).reduceByKey(add)
    missing = all_dict.subtractByKey(cat).map(lambda x: (x[0], 0)) #add 0 counts into dataset
    cat = cat.union(missing).map(lambda x: (x[0], x[1])) # add one to avoid 0
    return cat

def get_prob(x, cat_count):
    '''
    Get conditional probabilities for each word in a category. Add one to avoid 0.
    '''
    x = add_missing(x, super_vocab).map(lambda x: (x[0], (x[1] + 1) / (cat_count + TOTAL_WORD.value)))   #super_vocab is a global variable
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
    print (prior_prob)
    prior_values = np.array([math.log(prior_prob['CCAT']), math.log(prior_prob['ECAT']), math.log(prior_prob['GCAT']), math.log(prior_prob['MCAT'])], dtype='float128')  #make sure the order of the dict is correct.
    print (prior_values)

    all_count = word_count_all(all_text) # Count all the words


    test_path = os.path.join(script_dir, 'X_test_small.txt')
    test_text = spark.sparkContext.textFile(test_path)
    test_text = word_count_test(test_text)
    super_vocab = test_text.flatMap(lambda x: ([v, 1] for v in x)).union(all_count).reduceByKey(add) #super_vocab is the vocab in both training and testing
    TOTAL_WORD = spark.sparkContext.broadcast(super_vocab.map(lambda x: x[0]).count()) #broadcast the total word count value
    print (TOTAL_WORD.value)

    ccat = word_count_cat('CCAT', all_text)
    ecat = word_count_cat('ECAT', all_text)
    gcat = word_count_cat('GCAT', all_text)
    mcat = word_count_cat('MCAT', all_text)

    CCAT_COUNT = spark.sparkContext.broadcast(ccat.count())
    ECAT_COUNT = spark.sparkContext.broadcast(ecat.count())
    GCAT_COUNT = spark.sparkContext.broadcast(gcat.count())
    MCAT_COUNT = spark.sparkContext.broadcast(mcat.count())

    # get the probabilities for each category
    ccat = get_prob(ccat, CCAT_COUNT.value)
    ecat = get_prob(ecat, ECAT_COUNT.value)
    gcat = get_prob(gcat, GCAT_COUNT.value)
    mcat = get_prob(mcat, MCAT_COUNT.value)

    total_prob = ccat.union(ecat).union(gcat).union(mcat).groupByKey().mapValues(list)  #union all categories together

    total_prob = total_prob.map(lambda x: (x[0], [math.log(i) for i in x[1]]))
    #put the result into a dictionary. Key is each word and the value is 4-place tuple (which gives the conditional probability of a word given a category).
    total_prob_dict = total_prob.collectAsMap()
    TOTAL_PROB_GLOBAL = spark.sparkContext.broadcast(total_prob_dict)


    # Read the test file and get the results

    test_prob = test_text.map(lambda x: [TOTAL_PROB_GLOBAL.value[i] for i in x if i in total_prob_dict])
    test_prob = test_prob.collect()
    test_prob = np.array([reduce(lambda x, y: np.add(x, y), np.array(i, dtype = 'float128')) for i in test_prob]) # calculate the prob for each doc given words in each doc

    test_prob = np.array([np.add(i, prior_values) for i in test_prob]) #multiply value of each doc with the prior prob of each category
    # print (test_prob)
    maxidx = np.argmax(test_prob, axis = 1)
    cat_dict = {0: 'CCAT', 1: 'ECAT', 2: 'GCAT', 3: 'MCAT'} # hand-code dict type. The order is based on the order of union operations earlier

    res = [cat_dict[i] for i in maxidx]

    # Testing
    test_res_path = os.path.join(script_dir, 'y_test_small.txt')

    # with open(test_res_path, 'w') as writefile:
    #     writefile.write('\n'.join(res))


    #For comparing results
    with open(test_res_path, 'r') as readFile:
        res_label = readFile.read()
    res_label = res_label.splitlines()
    accu = 0
    for i, v in enumerate(res):
        if v in res_label[i]:
           accu += 1
    print (accu/len(res))
    #
    #
    # Todo -- change all num into "NUM"
