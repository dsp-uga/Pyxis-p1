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
    y_test = sc.textFile(args.y_test)

    stopwords = []
    if args.stopwords_path != None:
        text_file = open(args.stopwords_path, "r")
        for line in text_file.readlines():
            stopwords.append(line.strip())
    stopwords_rdd = sc.broadcast(stopwords)



    #pre-processing x-train
    preprocessed_text = X_Preprocessing(x_train, min_word_length)
    #pre-processing x-test
    test_text = X_Preprocessing(x_test, min_word_length)
    #pre-processing y-train
    preprocessed_label = y_Preprocessing(y_train)

    # sc = SparkContext.getOrCreate()
    all_label, all_text_label = process_label_text(preprocessed_label, preprocessed_text) #get all lable RDD and all training text with label RDD
    LABEL_COUNT = spark.sparkContext.broadcast(len(all_label.collect()))  #broadcast all label counts
    ALL_PRIOR = count_label(all_label, LABEL_COUNT.value).collectAsMap()l
    TOTAL_VOCAB = get_total_vocab(preprocessed_text, test_text)#super_vocab is the vocab in both training and testing
    TOTAL_VOCAB_COUNT = spark.sparkContext.broadcast(total_vocab.map(lambda x: x[0]).count()) #broadcast the total word count value


    ccat = word_count_cat('CCAT', preprocessed_text)
    ecat = word_count_cat('ECAT', preprocessed_text)
    gcat = word_count_cat('GCAT', preprocessed_text)
    mcat = word_count_cat('MCAT', preprocessed_text)

    # create map from words to probabilities
    TOTAL_WORD_PROB = get_total_word_prob(ccat, ecat, gcat, mcat).collectAsMap()

    # replace words in documents with their probabilities
    document_word_probs = words_to_probs(x_test,TOTAL_WORD_PROB)
    # aggregate total probabilities for each document
    document_total_probs = docs_to_probs(document_word_probs, ALL_PRIOR)
    # get index of maximum probability (which corresponds to a class)
    predictions = class_preds(document_total_probs)
    # compare calculated maximum probability class with actual label to report accuracy
    print(accuracy(predictions, y_test))