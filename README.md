sparkgram
=========

A simple package for processing text corpora with Spark. Extracts ngrams,
produces vector representations of documents in a corpus etc.

Install:
--------

First install [Spark](http://spark.apache.org). Then:

```
> git clone https://github.com/rokroskar/sparkgram.git
> cd sparkgram
> python setup.py install
```
Start your spark cluster etc. and launch a pyspark shell.

From the pyspark shell:

```
>>> import sparkgram, glob
>>> dv = sparkgram.SparkDocumentVectorizer(glob.glob('*txt'))
>>> dv.docvec_rdd.take(5)
```

This will return the vector representations of the first five documents. 
