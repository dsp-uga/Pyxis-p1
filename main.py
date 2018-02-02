from operator import add
import glob
import os
from argparse import ArgumentParser
from pyspark import SparkContext
import re
import math
import unittest
from pre_processing import *
from training import *
from testing import *
import math
import numpy as np



if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    parser = ArgumentParser()
    parser.add_argument("-x", "--x", dest="x_train", help="Give the path for x_train.", required = True)
    parser.add_argument("-y", "--y", dest="y_train", help="Give the path for y_train.", required = True)
    parser.add_argument("-xtest", "--xtest", dest="x_test", help="Give the path for x_test.", required = True)
    parser.add_argument("-st", "--stopwords", dest="stopwords_path", help="Give the path for stopwords.[DEFAULT: \".\"]", default = None)
    parser.add_argument("-l", "--len", dest="min_word_length", help="Specify the minimum length for words.[DEFAULT: 2]", default = 2)
    args = parser.parse_args()


    min_word_length = int(args.min_word_length)
    x_train = sc.textFile(args.x_train)
    y_train = sc.textFile(args.y_train)
    x_test = sc.textFile(args.x_test)
    # y_test = sc.textFile(args.y_test)

    stopwords = []
    if args.stopwords_path != None:
        text_file = open(args.stopwords_path, "r")
        for line in text_file.readlines():
            stopwords.append(line.strip())
    stopwords_rdd = sc.broadcast(stopwords)



    #pre-processing x-train
    preprocessed_text = X_Preprocessing(x_train, min_word_length, stopwords_rdd)
    print(preprocessed_text.take(1))
    #pre-processing x-test
    test_text = X_Preprocessing(x_test, min_word_length, stopwords_rdd)
    #pre-processing y-train
    preprocessed_label = y_Preprocessing(y_train)

    #adding n-grams
    preprocessed_text = add_n_grams(2, preprocessed_text)
    test_text = add_n_grams(2, test_text)

    # sc = SparkContext.getOrCreate()
    all_label, all_text_label = process_label_text(preprocessed_label, preprocessed_text) #get all lable RDD and all training text with label RDD
    LABEL_COUNT = sc.broadcast(len(all_label.collect()))  #broadcast all label counts
    ALL_PRIOR = count_label(all_label, LABEL_COUNT.value).collectAsMap()
    # print (len(preprocessed_text.collect()[1]))
    TOTAL_VOCAB = get_total_vocab(preprocessed_text, test_text)#super_vocab is the vocab in both training and testing

    # print (len(total_vocab.collect()))
    TOTAL_VOCAB_COUNT = sc.broadcast(TOTAL_VOCAB.count()) #broadcast the total word count value


    ccat = word_count_cat('CCAT', all_text_label)
    ecat = word_count_cat('ECAT', all_text_label)
    gcat = word_count_cat('GCAT', all_text_label)
    mcat = word_count_cat('MCAT', all_text_label)


    # create map from words to probabilities
    TOTAL_WORD_PROB = get_total_word_prob(ccat, ecat, gcat, mcat, TOTAL_VOCAB_COUNT, TOTAL_VOCAB).collectAsMap()
    # replace words in documents with their probabilities
    document_word_probs = words_to_probs(test_text.zipWithIndex().map(lambda x: (x[1], x[0])),TOTAL_WORD_PROB)

    # aggregate total probabilities for each document
    document_total_probs = docs_to_probs(document_word_probs, ALL_PRIOR)

    # get index of maximum probability (which corresponds to a class)
    cat_dict = {0: 'CCAT', 1: 'ECAT', 2: 'GCAT', 3: 'MCAT'}
    predictions = class_preds(document_total_probs).map(lambda x : cat_dict[x[1]]).collect()
    # compare calculated maximum probability class with actual label to report accuracy


    # with open(test_res_path, 'w') as writefile:
    #     writefile.write('\n'.join(res))
    #
    #
    #For comparing results
    with open('/Users/yuanmingshi/downloads/prxis-p1/y_test_small.txt', 'r') as readFile:
        res_label = readFile.read()
    res_label = res_label.splitlines()
    accu = 0
    for i, v in enumerate(predictions):
        if v in res_label[i]:
           accu += 1
    print (accu/len(res_label))
