"""
Document Vectorizer
===================

includes classes for vectorizing documents in a text corpus using Spark.
"""

import re
import sys, glob, os
from random import shuffle
import mmh3
import numpy as np
from pyspark.mllib.linalg import SparseVector
from collections import defaultdict
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

homedir = os.environ['HOME']


# define some simple tokenizers
alphanum_regexp = re.compile(ur'(?u)\b\w\w+\b', re.UNICODE)
alphanum_tokenizer = lambda doc : alphanum_regexp.findall(doc)

alpha_regexp = re.compile(ur"(?u)\b[A-Za-z']+\b", re.UNICODE)
alpha_tokenizer = lambda doc : alpha_regexp.findall(doc)


# define NLTK stemmer
class StemTokenizer(object):
    def __init__(self):
        self.snowball = SnowballStemmer("english")
    def __call__(self, doc):
        doc = re.sub("[0-9][0-9,]*", "_NUM", doc)
        return [self.snowball.stem(t) for t in word_tokenize(doc) if re.match('\w\w+$',t)]


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
                          stop_words = None, tokenizer = alpha_tokenizer) :
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
    for ngram in word_ngrams(tokenizer(text), ngram_range = ngram_range, stop_words=stop_words):
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
    for ngram in word_ngrams(tokenizer(text),
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
                 num_partitions = None, features_max = None, tokenizer = alpha_tokenizer) :

        self._sc = sc
        self._ngram_range = ngram_range
        self._stop_words = stop_words
        self._nmin = nmin
        self._nmax = nmax
        self._num_partitions = num_partitions
        self._doclist = doclist
        self._features_max = features_max if features_max is not None else 2**31
        self._tokenizer = tokenizer

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

        num_partitions, ngram_range, sw, tokenizer = self._num_partitions, self._ngram_range, self._stop_words, self._tokenizer

        # generate an RDD of (ngram,context) pairs
        ng_inv = self.doc_rdd.flatMap(lambda (x,y):
                                    [(ngram,x) for ngram in word_ngrams(tokenizer(y), ngram_range, sw)])

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

        freq_rdd = self.get_corpus_frequency_rdd()

        return freq_rdd.filter(lambda (_,count): count < nmax and count > nmin) \
                       .sortByKey()\
                       .map(lambda (x,_): x)


    def get_corpus_frequency_rdd(self) :
        """
        Return an RDD of ``(key,value)`` pairs, where ``key`` is the ngram and
        key is the number of documents that ngram appears in throughout the corpus.
        """
        vocab_rdd = self.ngram_rdd.flatMap(lambda (_,x): [y[0] for y in x])

        return vocab_rdd.map(lambda x: (x,1)).reduceByKey(lambda a,b : a+b, self._num_partitions).sortByKey()


    def reset(self) :
        """
        Discard the calculated RDDs, i.e. ngram_rdd, vocab_rdd, and docvec_rdd
        """
        del(self._ngram_rdd)
        del(self._vocab_rdd)
        del(self._docvec_rdd)

        self._ngram_rdd, self._vocab_rdd, self._docvec_rdd = None, None, None


    @staticmethod
    def read_docs(doc) :
        with open(doc) as f:
            text = f.read().lower()

        return (doc,text)

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
            doclist = self._doclist
            self._doc_rdd = self._sc.parallelize(doclist) \
                                    .map(SparkDocumentVectorizer.read_docs)
        return self._doc_rdd


    @property
    def ngram_rdd(self) :
        """
        Transform the text into [(ngram, ID), count] pairs
        """
        ngram_range = self._ngram_range
        stop_words = self._stop_words
        features_max = self._features_max
        tokenizer = self._tokenizer

        if self._ngram_rdd is None :
            if self._nmin is None and self._nmax is None :
                self._ngram_rdd = self.doc_rdd.mapValues(
                    lambda x: ngram_frequency(x, ngram_range,
                                              stop_words, tokenizer, features_max))
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


    def get_vocab_map(self) :
        """
        Return an RDD of (ngram, hash) pairs
        """
        features_max = self._features_max

        return self.vocab_rdd.map(lambda x: (x,(mmh3.hash(x) & 0x7FFFFFFF) % features_max))

    @staticmethod
    def _write_single_partition_matrix(id, iterator, counts, datalen, path, filename, format):
        """
        Output the feature vectors from this partition into a file.
        The matrix has to fit into worker memory.
        """
        import numpy as np
        import scipy as sp
        from scipy.sparse import csr_matrix

        num_vectors = counts[id]
        mydatalen = datalen[id]

        # initialize data arrays
        values = np.empty(mydatalen,'int32')
        indices = np.empty(mydatalen,'int32')
        lengths = np.empty(num_vectors,'int32')
        indptr = np.zeros(num_vectors+1, 'int32')
        contexts = np.ndarray(counts[id],dtype=np.object)

        # get the data from the SparseVectors
        n = 0
        for i,(context, vec) in enumerate(iterator) :
            l = len(vec.indices)
            indices[n:n+l] = vec.indices
            values[n:n+l] = vec.values
            contexts[i] = context
            lengths[i] = len(vec.values)
            n += l

        indptr[1:] = np.cumsum(lengths)

        filename = '%s/%s_%d'%(path,filename,id)

    	if format == 'numpy' :
        	# write the numpy arrays to a file with
        	np.savez('%s.npz'%(filename),values = values, indices = indices,
                     indptr = indptr, contexts = contexts, datalen = mydatalen,
                     num_vectors = num_vectors)

        elif format == 'scipy' :
            # output using matrix format
            mat = csr_matrix((values,indices,indptr), shape = (len(iterator),vec.size))
            scipy.io.mmwrite(filename,mat)
        else :
            raise RuntimeError("must specify format as either numpy or scipy")

        yield 1


    def write_feature_matrix(self, path, filename = 'docvec_data', format = 'numpy'):
        """
        Output the feature matrix into a file. The matrices
        on individual workers have to fit into memory.
        """
        # cache this RDD in case it isn't already
        self.docvec_rdd.cache()

        # number of elements per partitions
        counts = get_partition_counts(self.docvec_rdd)

        # number of nonzero elements in all of the vectors
        datalen = nonzero_vector_elements(self.docvec_rdd)

        res = self.docvec_rdd.mapPartitionsWithIndex(
                   lambda id, iterator: \
                   SparkDocumentVectorizer\
                   ._write_single_partition_matrix(id, iterator, counts, datalen,
                                                   path, filename, format)).sum()

        print 'created %d files %s/%s_[0-%d]'%(res,path,filename,res)


def load_feature_matrix(path, filename = 'docvec_data', format = 'numpy') :
    """Create a scipy sparse matrix from partition data saved on disk"""
    import glob
    from scipy.sparse import csr_matrix

    files = glob.glob('%s/%s_*.npz'%(path,filename))
    nparts = len(files)

    # open up the data files
    data = []
    start_indices = np.zeros(nparts,dtype='int32')
    total_datalen = 0
    total_vectors = 0

    # get the individual lengths and element offsets
    for i,dat in enumerate((np.load(f) for f in files)) :
        data.append(dat)
        if i > 0 : start_indices[i] = data[i-1]['datalen']
        total_datalen += dat['datalen']
        total_vectors += dat['num_vectors']

    start_indices = np.cumsum(start_indices)

    values = np.ndarray(total_datalen,dtype='int32')
    indices = np.ndarray(total_datalen,dtype='int32')
    indptr = np.ndarray(total_vectors+1,dtype='int32')
    contexts = np.ndarray(total_vectors,dtype=np.object)

    nval = 0
    nptr = 1 # the first item is zero
    for i,dat in enumerate(data) :
        lenval = dat['datalen']
        lenptr = dat['num_vectors']

        values[nval:nval+lenval] = dat['values']
        indices[nval:nval+lenval] = dat['indices']
        indptr[nptr:nptr+lenptr] = dat['indptr'][1:] + start_indices[i]
        contexts[nptr-1:nptr+lenptr-1] = dat['contexts']

        nval += lenval
        nptr += lenptr

    np.savez('%s/%s-all'%(path,filename),contexts=contexts, values=values, indices=indices, indptr=indptr)

    return csr_matrix((values,indices,indptr))

### Some handy utility functions and stuff that is not used anymore
def nonzero_vector_elements(rdd) :
    """Returns non-zero element count on each partition"""
    def helper(id,iterator):
        yield (id,sum([len(vec.values) for (context,vec) in iterator]))
    return rdd.mapPartitionsWithIndex(helper).collectAsMap()

def get_partition_counts(rdd) :
    """Returns partition indices and item counts"""
    return rdd.mapPartitionsWithIndex(_count_partitions).collectAsMap()


def _count_partitions(id,iterator):
    c = sum(1 for _ in iterator)
    yield (id,c)


def _zip_with_index(l,indices,k) :
    start_ind = sum(indices[:k+1])
    for i, item in enumerate(l) :
        yield (item,start_ind+i)
