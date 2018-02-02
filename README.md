# Group Pyxis - Project 1 Scalable Document Classification
This repository classify documents by using Reuters Corpus, which is a set of news stories with multiple class labels. with Spark's Python API completed for CSCI8360: Data Science Practicum at the University of Georgia. By using Reuters Corpus, which is a set of new stories with multiple class labels. The training data is over 1 gigabyte and the testing data is roughly 117 MB with over 80000 documents/news stories.

These different news stories are split into different categories. In this project, we are only focusing on these four labels:
1. CCAT: Corporate/Industrial
2. ECAT: Economics
3. GCAT: Government/Social
4. MCAT: Markets

For documents with more than one label, we treat it as if it's observed once for each `CAT` label. In prediction, we only predict one label for each document.

## Getting started
### Prerequisites

This project uses Apache Spark. You'll need to have Spark installed on the target cluster. The SPARK_HOME environment variable should be set, and the Spark binaries should be in your system path. You also need to install the library of NLTK for stemming.

### Built with

- [Python 3.6](https://www.python.org/)
- [Apache Spark](https://spark.apache.org/)
- [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html/)
- [Numpy](https://docs.scipy.org/doc/numpy-1.13.0/)
- [NLTK](http://www.nltk.org/)
- [Google Cloud Platform](https://cloud.google.com)

### How to run
To get it running, you can use
```
spark-submit main.py **kwargs
```
to run the main program and the output should be in the path you specified.
There are several keyword arguments for the program. They are as follows:
- `-x`: the path for the x_training file (documents). REQUIRED
- `-y`: the path for the y_training file (labels). REQUIRED
- `-xtest`: the path for the x_testing file. (documents) REQUIRED
- `-st`: the path for the stopword files. OPTIONAL, default value None.
- `-l`: length of words to throw away. OPTIONAL, default value 2 (i.e. ignore all words with length 2 or less).
- `-o`: path for the output file `output.txt`. OPTIONAL, default value is the same as the main.py file.

If you find any problem, please create a ticket!

### Specific features
Here are some methods that we use in this project:
- Baseline: Naive Base Model
- removal of words whose lengths are less than or equal to 2
- removal of stop words: we choose a long list of stop words (see [here](https://www.ranks.nl/stopwords) and find "a very long stopword list").
- TF-IDF (term frequency inverse document frequency)
- n-gram (2-gram; 3-gram or higher could be achieved by tuning the parameter in `pre_processing.py`)
- stemming: we use NLTK's porter stemmer (see http://www.nltk.org/howto/stem.html)

## Contributors (alphabetically sorted)
- Layton Hayes, Institute of Artificial Intelligence, University of Georgia
- Parya Jandaghi, Department of Computer Science, University of Georgia
- Jeremy Shi, Institute of Artificial Intelligence, University of Georgia

See the [contributors](./CONTRIBUTORS.md) file for detailed contributions.
We also thank [Shannon Quinn](http://magsol.github.io/) for helpful instructions.

## License
MIT

## TODO
- Tuning the stopword list to improve the accuracy.
- Improve smoothing in tf-idf.
