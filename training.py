from operator import add
import glob
import os
from argparse import ArgumentParser
from pyspark import SparkContext
import re
import math


def process_label_text(label, text):
    '''
    Process training texts and labels and return word counts. Labels are only with CAT;
    Return a list with all labels and texts
    '''
    label = label.zipWithIndex().map(lambda x: [x[1], x[0]])   #Use zipWithIndex to get the indices and swap value and key
    text = text.zipWithIndex().map(lambda x: [x[1], x[0]])     #Use zipWithIndex to get the indices and swap value and key
    text = text.join(label).sortByKey().map(lambda x: x[1])    #Join texts and labels based on the key; sort the list, and delete the key column
    text = text.flatMap(lambda x: ((x[0], i) for i in x[1]))   #Remove redundant labels and flatten texts with multiple labels
    label = text.map(lambda x: x[1])  #return all label types
    return label, text
    #Count the numbers of total texts and calculate the prior probability of each

def add_missing(cat, all_dict):
    '''
    Not every document (or every type of document) has every word in the vocab. So we need to add the word that are missing and put 0 count for them.
    '''
    cat = cat.flatMap(lambda x: ([v, 1] for v in x)).reduceByKey(add)
    missing = all_dict.subtractByKey(cat).map(lambda x: (x[0], 0)) #add 0 counts into dataset
    cat = cat.union(missing).map(lambda x: (x[0], x[1])) # add one to avoid 0
    return cat

def count_label(label, label_count):
    '''
    Count the number of the all labels and the prior probability for each category; label_word_count is the count of all labels.
    Return a RDD with ('CATEGORY_NAME', "LOG_PROB_OF_THE_CATEGORY") tuples as elements.
    '''
    all_prior = label.map(lambda x: (x, 1)).reduceByKey(add).map(lambda x: (x[0], math.log(x[1]/label_count)))
    return all_prior

def get_prob(x, cat_count):
    '''
    Get conditional probabilities for each word in a category. Add one for smoothing.
    cat_count is how many words are in each category
    total_vocab is a RDD like this --- [['word1', 2], ['word2', 3], ['word3', 1]], which is the total counts in both training and testing.
    t_v_count is size (type: number) of the vocabulary (including both training and testing.)
    '''
    x = x.map(lambda x: (x[0], (x[1]) / (cat_count)))   #super_vocab is a global variable
    return x

def get_total_word_prob(cat1, cat2, cat3, cat4, t_v_count, total_vocab):
    '''
    For each word, return the conditional probability of being each category of document. Each cat is a RDD;
    Four arguments are lists of words_in_a_doc for each category (i.e. each element of the argument is a list with all words with repeat from that category).
    '''
    ccat = get_prob(cat1, cat1.flatMap(lambda x: (v for v in x)).reduce(lambda x, y: x[1] + y[1]))
    ecat = get_prob(cat2, cat2.flatMap(lambda x: (v for v in x)).reduce(lambda x, y: x[1] + y[1]))
    gcat = get_prob(cat3, cat3.flatMap(lambda x: (v for v in x)).reduce(lambda x, y: x[1] + y[1]))
    mcat = get_prob(cat4, cat4.flatMap(lambda x: (v for v in x)).reduce(lambda x, y: x[1] + y[1]))
    total_word_prob = ccat.union(ecat).union(gcat).union(mcat).groupByKey().mapValues(list)  #union all categories together
    total_word_prob = total_word_prob.map(lambda x: (x[0], [math.log(i) for i in x[1]]))
    return total_word_prob

def get_total_vocab(training_text, testing_text):
    '''
    getting total vocab in both training and testing docs, for the purpose of smoothing). Two arguments are preprocessed training and testing texts.
    '''
    return testing_text.flatMap(lambda x: ([v, 1] for v in x)).union(training_text.flatMap(lambda x: ([v, 1] for v in x))).reduceByKey(add)

def word_count_cat(cat_name, rdd):
    '''
    Return word count per category among all text. cat_name is the name of the category (type:str).
    Input rdd is the collection of all texts with corresponding labels(type: RDD).
    '''
    return rdd.filter(lambda x: x[1] == cat_name).map(lambda x: x[0])

if __name__ == '__main__':

    # sc = SparkContext.getOrCreate()
    all_label, all_text_label = process_label_text(preprocessed_label, preprocessed_text) #get all lable RDD and all training text with label RDD
    LABEL_COUNT = spark.sparkContext.broadcast(len(all_label.collect()))  #broadcast all label counts
    ALL_PRIOR = count_label(all_label, LABEL_COUNT.value)
    TOTAL_VOCAB = get_total_vocab(preprocessed_text, test_text)#super_vocab is the vocab in both training and testing
    TOTAL_VOCAB_COUNT = spark.sparkContext.broadcast(total_vocab.map(lambda x: x[0]).count()) #broadcast the total word count value


    ccat = word_count_cat('CCAT', preprocessed_text)
    ecat = word_count_cat('ECAT', preprocessed_text)
    gcat = word_count_cat('GCAT', preprocessed_text)
    mcat = word_count_cat('MCAT', preprocessed_text)

    TOTAL_WORD_PROB = get_total_word_prob(ccat, ecat, gcat, mcat)

# get the probabilities for each category
