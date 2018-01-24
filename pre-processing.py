
from operator import add
import glob
import os
from argparse import ArgumentParser
from pyspark import SparkContext
<<<<<<< HEAD
=======
import re
>>>>>>> pre-processing

sc = SparkContext.getOrCreate()

parser = ArgumentParser()
parser.add_argument("-x", "--x", dest="x_train", help="Give the path for x_train.")
parser.add_argument("-y", "--y", dest="y_train", help="Give the path for y_train.")
parser.add_argument("-xtest", "--xtest", dest="x_test", help="Give the path for x_test.")
parser.add_argument("-ytest", "--ytest", dest="y_test", help="Give the path for y_test.")
parser.add_argument("-st", "--stopwords", dest="stopwords_path", help="Give the path for stopwords.")
parser.add_argument("-l", "--len", dest="min_word_length", help="Specify the minimum length for words.")
#parser.add_argument("-o", "--out", dest="output_path", help="Give the path for output.")
args = parser.parse_args()


min_word_length = int(args.min_word_length)
x_train = sc.textFile(args.x_train)
y_train = sc.textFile(args.y_train)
x_test = sc.textFile(args.x_test)
y_test = sc.textFile(args.y_test)

text_file = open(args.stopwords_path, "r")
stopwords = []
for line in text_file.readlines():
    stopwords.append(line.strip())
stopwords_rdd = sc.broadcast(stopwords)

#ignore everything except alphabet
<<<<<<< HEAD
def AllAlpha(line):
    new_line = []
    for word in line:
        if word.isalpha():
            new_line.append(word)
=======
def RemoveEcxeptAlphabets(line):
    regex = re.compile('[^a-zA-Z]')
    new_line = []
    for word in line:
        word = regex.sub('', word)
        new_line.append(word)
>>>>>>> pre-processing
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





#pre-processing x-train
text = x_train.map( lambda line : line.split(" "))\
.map(lambda line: [word.lower() for word in line])\
<<<<<<< HEAD
.map(lambda line: AllAlpha(line))\
=======
.map(lambda line: RemoveEcxeptAlphabets(line))\
>>>>>>> pre-processing
.map(lambda line: MinimumLength(line, 2))\
.map(lambda line: NotStopWords(line, stopwords))




<<<<<<< HEAD



#pre-processing x-test
test_text = x_test.map( lambda line : line.split(" "))\
.map(lambda line: [word.lower() for word in line])\
.map(lambda line: AllAlpha(line))\
=======
#pre-processing x-test
test_text = x_test.map( lambda line : line.split(" "))\
.map(lambda line: [word.lower() for word in line])\
.map(lambda line: RemoveEcxeptAlphabets(line))\
>>>>>>> pre-processing
.map(lambda line: MinimumLength(line, 2))\
.map(lambda line: NotStopWords(line, stopwords))






#pre-processing y-train
labels = y_train.map( lambda label : label.split(",") )
labels = labels.map(lambda label : [lb for lb in label if lb[-3:]=='CAT'])





#pre-processing y-test
test_labels = y_test.map( lambda label : label.split(",") )
test_labels = test_labels.map(lambda label : [lb for lb in label if lb[-3:]=='CAT'])

