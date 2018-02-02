
from operator import add
import glob
import os
from argparse import ArgumentParser
from pyspark import SparkContext
import re
from nltk.stem.porter import *

from nltk.stem.snowball import SnowballStemmer


if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    stemmer = PorterStemmer()

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
    text = X_Preprocessing(x_train, min_word_length)
    #pre-processing x-test
    test_text = X_Preprocessing(x_test, min_word_length)
    #pre-processing y-train
    labels = y_Preprocessing(y_train)
#ignore everything except alphabet
def RemoveEcxeptAlphabets(line):
    regex = re.compile('[^a-zA-Z]')
    new_line = []
    for word in line:
        word = regex.sub('', word)
        new_line.append(word)
    return new_line


#ignore those words with the length less than minimum length
def MinimumLength(line, minimum_length):
    new_line = []
    for word in line:
        if len(word) > minimum_length:
            new_line.append(word)
    return new_line


#ignore the words appear in stopwords file
def NotStopWords(line, stopwordsList):
    new_line = []
    for word in line:
        if not word in stopwordsList:
            new_line.append(word)
    return new_line


def add_n_grams(n, documents):
    res_rdd = documents
    for i in range(n+1):
        if(i < 2): continue
        i_grams = documents.map(lambda doc: n_grams(i, doc))
        res_rdd = res_rdd.zip(i_grams).map(lambda x: x[0]+x[1]) #add the new grams to res_rdd
    return res_rdd

def n_grams(n, document):
    words = document
    for i in range(n):
        if i==0: continue
        words = words[:-1]
        addition = document[i:]
        words = [x[0]+" "+x[1] for x in zip(words,addition)]
    return words


#make all words lowercase
#remove everything in words except alphabets
#remove words with the length less than minimum length
#remove words appear in stopwords
def X_Preprocessing( text, minimum_length, stopwords_rdd):
    stemmer = PorterStemmer()
    return text.map( lambda line : line.split(" "))\
    .map(lambda line: [word.lower() for word in line])\
    .map(lambda line: RemoveEcxeptAlphabets(line))\
    .map(lambda line: MinimumLength(line, minimum_length))\
    .map(lambda line: NotStopWords(line, stopwords_rdd.value))\
    .map(lambda line: [stemmer.stem(word) for word in line])

#remove those labels are not ending with "CAT"
def y_Preprocessing(labels):
    return labels.map( lambda label : label.split(",") )\
    .map(lambda label : [lb for lb in label if lb[-3:]=='CAT'])
