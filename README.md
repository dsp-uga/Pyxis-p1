# Group Pyxis - Project 1
This repository classify documents by using Reuters Corpus, which is a set of news stories with multiple class labels. with Spark's Python API completed for CSCI8360: Data Science Practicum at the University of Georgia. By using Reuters Corpus, which is a set of new stories with multiple class labels. \\
Different news stories are split into different categories. In this project, we are only focusing on these four labels:
1. CCAT: Corporate/Industrial
2. ECAT: Economics
3. GCAT: Government/Social
4. MCAT: Markets

For documents with more than one label, we treat it as if it's observed once for each `CAT` label.

### Prerequisites

This project uses Apache Spark. You'll need to have Spark installed on the target cluster. The SPARK_HOME environment variable should be set, and the Spark binaries should be in your system path. You also need to install the library of NLTK for stemming.

### Built with

- [Python 3.6](https://www.python.org/)
- [Apache Spark](https://spark.apache.org/)
- [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html/)

### How to run
You can use `spark-submit main.py **kwargs` to run the main program and the output should be in the path you specified.
There are several keyword arguments for the program. They are as follows:
- `-x`: the path for the x_training file (documents). REQUIRED
- `-y`: the path for the y_training file (labels). REQUIRED
- `-xtest`: the path for the x_testing file. (documents) REQUIRED
- `-st`: the path for the stopword files. OPTIONAL, default value None.
- `-l`: length of words to throw away. OPTIONAL, default value 2 (i.e. ignore all words with length 2 or less).

If you find any problem, please create a ticket!
## Authors (alphabetically sorted)
- Layton Hayes
- Parya Jandaghi
- Jeremy Shi

See the [contributors](./CONTRIBUTORS.md) file for detailed contributions.
