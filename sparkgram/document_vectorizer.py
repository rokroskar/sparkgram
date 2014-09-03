"""
Document Vectorizer
===================

includes classes for vectorizing documents in a text corpus using Spark.
"""

import re
import sys, glob, os
from random import shuffle
import mmh3
from pyspark.mllib.linalg import SparseVector
from collections import defaultdict

homedir = os.environ['HOME']


# define some simple tokenizers
alphanum_regexp = re.compile(ur'(?u)\b\w\w+\b', re.UNICODE)
alphanum_tokenizer = lambda doc : alphanum_regexp.findall(doc)

alpha_regexp = re.compile(ur"(?u)\b[A-Za-z']+\b", re.UNICODE)
alpha_tokenizer = lambda doc : alpha_regexp.findall(doc)


###########################
#
# FUNCTIONS THAT RUN ON RDD
#
###########################

def analyze(text) :
    return word_ngrams(tokenizer(text))


def word_ngrams(tokens, ngram_range=[1,1], stop_words=None):
    """
    Turn tokens into a sequence of n-grams after stop words filtering

    **Inputs**:

    *tokens*: a list of tokens

    **Optional Keywords**:

    *ngram_range*: a tuple with min, max ngram ngram_range

    *stop_words*: a list of stop words to use

    **Output**

    Generator yielding a list of ngrams in the desired range
    generated from the input list of tokens

    """
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    n_tokens = len(tokens)
    for n in xrange(min_n, min(max_n + 1, n_tokens + 1)):
        for i in xrange(n_tokens - n + 1):
            yield " ".join(tokens[i: i+n])


def ngram_vocab_frequency(vocab, text, ngram_range = [1,1],
                          stop_words = None, tokenier = alpha_tokenizer) :
    """
    Count the frequency of ngrams appearing in document ``text``
    matched against the ngrams stored in ``vocab``

    **Input**:

    *vocab*: is a dictionary built from the key, value pairs in the RDD

    *text*: raw string

    **Optional keywords**:

    *ngram_range*: a tuple with min, max ngrams to generate

    *stop_words*: a list of stop words to use

    *tokenizer*: function that will turn the raw text into tokens


    This is based on
    sklearn.feature_extraction.CountVectorizer._count_vocab

    """
    d = defaultdict(int)

    # count the occurences
    for ngram in word_ngrams(alpha_tokenizer(text), ngram_range = ngram_range, stop_words=stop_words):
        if ngram in vocab :
            d[ngram] += 1

    # extract the results into a list of tuples and sort by feature index
    res = [(vocab[ngram],d[ngram]) for ngram in d.keys()]
    res.sort()

    return res

def ngram_frequency(text, ngram_range=[1,1], stop_words = None,
                    tokenizer = alpha_tokenizer, feature_max = 2**32) :
    """
    Count the frequency of ngrams appearing in document ``text``
    by using string hashes.

    **Input**

    *text*: raw text to process

    **Optional keywords**

    *ngram_range*: a tuple with min, max ngrams to generate

    *stop_words*: a list of stop words to use

    *tokenizer*: function that will turn the raw text into tokens

    *feature_max*: maximum number of expected features. This sets the size of the
    sparse vectors generated during the document vectorization step and
    should be set to the nearest power of two larger than the expected
    number of features/ngrams

    **Output**

    a list of (ngram,count) tuples

    """
    from collections import defaultdict
    d = defaultdict(int)

    # count the occurences
    for ngram in word_ngrams(alpha_tokenizer(text), 
                             ngram_range = ngram_range, 
                             stop_words = stop_words) :
        d[ngram] += 1

    # extract the results into a list of tuples and sort by feature index
    vec = [(ngram,d[ngram]) for ngram in d.keys()]
    vec.sort()

    return vec


############################
#
# FUNCTIONS THAT RUN ON HOST
#
############################

def next_power_of_two(x) :
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x


class SparkDocumentVectorizer(object) :
    """
    General purpose document vectorizer to be used with Spark. This class
    provides the basic functionality like extracting ngrams from raw
    strings. You'll want to extend it to suit your needs and define
    the ``read_docs`` method and perhaps also the ``doc_rdd`` method if your
    data ingestion is more complicated.

    
    **Constructor inputs**:
    
    
    *sc* : Spark context
    
    *doclist*: list of documents to process
    
    **Optional Keywords**
    
    *ngram_range*: a tuple with min, max ngrams to generate
    
    *stop_words*: a list of stop words to use
    
    *nmin/nmax*: minimum occurence count to use in filtering
    
    *num_partitions*: the level of parallelism to use in spark for some
    of the processing steps
    
    *feature_max*: maximum number of expected features. This sets the size of the
    sparse vectors generated during the document vectorization step and
    should be set to the nearest power of two larger than the expected
    number of features/ngrams
    """
    
    
    def __init__(self, sc, doclist,
                 ngram_range = [1,1], stop_words = None, nmin = None, nmax = None,
                 num_partitions = None, features_max = 2**32) :

        self._sc = sc
        self._ngram_range = ngram_range
        self._stop_words = stop_words
        self._nmin = nmin 
        self._nmax = nmax 
        self._num_partitions = num_partitions
        self._doclist = doclist
        self._features_max = features_max

        # initialie the RDDs
        self._doc_rdd = None
        self._ngram_rdd = None
        self._vocab_rdd = None
        self._docvec_rdd = None


        # initialize other properties
        self._nfeatures = None


    def load_text(self) :
        """
        Load the document data and cache it in memory
        """
        self.doc_rdd.cache()
        self.doc_rdd.count()


    def transform(self) :
        """
        Generate the document feature vectors
        """
        return self.docvec_rdd

    
    def apply_filter(self, filter_rdd = None, filter_func = None, **kwargs) :
        """
        Applies the filtering to ngram_rdd  and regenerates the feature vectors. 

        The new ngram_rdd is built using only the ngrams contained in ``filter_rdd``.

        **Optional Keywords**
        
        *filter_rdd*: an RDD containing ngrams that will be used to construct a new ``ngram_rdd``
        
        *filter_func*: the function used to construct ``filter_rdd`` if ``filter_rdd`` is not given. 
                       default is :func:`SparkDocumentVectorizer.filter_vocab`

        **Keywords passed to ``filter_func``** 
        
        *nmin*: minimum occurence count

        *nmax*: maximum occurence count

        """

        if filter_func is None : 
            filter_func = self.filter_vocab

        if filter_rdd is None : 
            filter_rdd = filter_func(**kwargs)

        self._ngram_rdd = self.filter_by_rdd(filter_rdd)

        # docvec_rdd and vocab_rdd are both derived from ngram_rdd,
        # so force reevaluation
        del(self._docvec_rdd); self._docvec_rdd = None
        del(self._vocab_rdd); self._vocab_rdd = None        
    

    def filter_by_rdd(self, filt_rdd) : 
        """
        Return a filtered ngram RDD based on ``nmin`` and ``nmax`` occurences of words 
        in different documents.
        """
        
        num_partitions, ngram_range, sw = self._num_partitions, self._ngram_range, self._stop_words
        
        # generate an RDD of (ngram,context) pairs
        ng_inv = self.doc_rdd.flatMap(lambda (x,y): 
                                    [(ngram,x) for ngram in word_ngrams(alpha_tokenizer(y), ngram_range, sw)])
        
        # do a join between the filtered vocabulary and the (ngram,context) RDD
        filtered_ngrams = filt_rdd.map(lambda x: (x,None)).join(ng_inv, num_partitions)
        
        # invert the filtered ngram RDD to get (context, ngram) pairs, then group by context
        ngram_rdd = filtered_ngrams.map(lambda (x,y): (y[1], x)).groupByKey(num_partitions)

        # return the ngram_rdd but with counted occurences of ngrams
        return ngram_rdd.map(lambda (context,ngrams): SparkDocumentVectorizer.count_ngrams(context,ngrams))
        

    def filter_vocab(self, nmin = None, nmax = None) : 
        """
        Filter the vocabulary RDD by min,max occurence
        """
        if nmin is None: nmin = self._nmin if self._nmin is not None else 0
        if nmax is None: nmax = self._nmax if self._nmax is not None else sys.maxint

        num_partitions = self._num_partitions

        vocab_rdd = self.ngram_rdd.flatMap(lambda (_,x): [y[0] for y in x])

        return vocab_rdd.map(lambda x: (x,1))\
                        .reduceByKey(lambda a,b: a+b, num_partitions) \
                        .filter(lambda (_,count): count < nmax and count > nmin) \
                        .sortByKey()\
                        .map(lambda (x,_): x)


    def reset(self) :
        """
        Discard the calculated RDDs, i.e. ngram_rdd, vocab_rdd, and docvec_rdd
        """
        del(self._ngram_rdd)
        del(self._vocab_rdd)
        del(self._docvec_rdd)
        
        self._ngram_rdd, self._vocab_rdd, self._docvec_rdd = None, None, None


    @staticmethod
    def read_docs(sc, doclist) :
        if type(doclist) is not list :
            raise RuntimeError("Please supply a list of filenames for processing")
        
        return sc.parallelize(doclist)\
            .map(lambda x: open(x).read().lower())
    

    @staticmethod
    def count_ngrams(context, ngrams) : 
        d = defaultdict(int)

        for ngram in ngrams : 
            d[ngram] += 1
        
        return (context, [(ngram,d[ngram]) for ngram in d.keys()])

    #
    # RDD property definitions
    #

    @property
    def doc_rdd(self) :
        """
        RDD containing the raw text partitioned across the cluster
        """
        if self._doc_rdd is None :
            self._doc_rdd = SparkDocumentVectorizer.read_docs(self._sc, self._doclist)
        return self._doc_rdd


    @property
    def ngram_rdd(self) :
        """
        Transform the text into [(ngram, ID), count] pairs
        """
        ngram_range = self._ngram_range
        stop_words = self._stop_words
        features_max = self._features_max

        if self._ngram_rdd is None :
            if self._nmin is None and self._nmax is None : 
                self._ngram_rdd = self.doc_rdd.mapValues(
                    lambda x: ngram_frequency(x, ngram_range,
                                              stop_words, features_max))
            else : 
                self.apply_filter()

        return self._ngram_rdd
    

    @property
    def vocab_rdd(self) :
        """
        Extract the vocabulary from the ngram RDD
        """
        num_partitions, nmin, nmax = self._num_partitions, self._nmin, self._nmax
        
        if self._vocab_rdd is None :
            # if no occurence filtering is required, get the distinct ngrams
            if nmin is None and nmax is None :
                self._vocab_rdd = self.ngram_rdd.flatMap(lambda (_,x): [y[0] for y in x]) \
                                      .map(lambda x: (x,None)) \
                                      .reduceByKey(lambda x,_: x, num_partitions) \
                                      .sortByKey() \
                                      .map(lambda (x,_): x)

            # perform occurence filtering
            else :
                self._vocab_rdd = self.filter_vocab()

        return self._vocab_rdd


    @property
    def docvec_rdd(self) :
        """
        Extract the document feature matrix

        Resulting RDD is a list of (metadata, SparseVector) pairs, where
        metadata is some identifying feature of each document. This could
        be just a random index, or some other more meaningful data so that
        the list can later be turned into a (LabeledPoint, SparseVector)
        list that can be passed to MLlib, for example.
        """

        features_max = self._features_max

        if self._docvec_rdd is None :
            # The vectors are [[(metadata),[(ngram,ngram_ID),count],[...]]]
            # We want to have [[(metadata),SparseVector[(ngram_ID,count),...]]], i.e.
            # just IDs and counts, no ngram string
            self._docvec_rdd = self.ngram_rdd.mapValues(
                lambda x: SparseVector(
                    features_max,[((mmh3.hash(y[0]) & 0x7FFFFFFF) % features_max, y[1]) for y in x]))

        return self._docvec_rdd


    @property
    def nfeatures(self) :
        if self._nfeatures is None :
            self._nfeatures = self.vocab_rdd.count()
        return self._nfeatures


### Some handy utility functions and stuff that is not used anymore
def get_partition_counts(rdd) :
    """Returns partition indices and item counts"""
    return rdd.mapPartitionsWithSplit(_count_partitions).collectAsMap()


def _count_partitions(id,iterator):
    c = sum(1 for _ in iterator)
    yield (id,c)


def _zip_with_index(l,indices,k) :
    start_ind = sum(indices[:k+1])
    for i, item in enumerate(l) :
        yield (item,start_ind+i)
