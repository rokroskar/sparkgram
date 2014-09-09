import sparkgram
from sparkgram.document_vectorizer import SparkDocumentVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os, sys
import numpy as np

short_doclist = ['%s/testdata/short_test%d'%(os.getcwd(),i+1) for i in range(4)]

def setup() :
    global cv, dv, sc

    # scikit-learn CountVectorizer
    cv = CountVectorizer('filename',tokenizer=sparkgram.document_vectorizer.alpha_tokenizer,
                         ngram_range = [1,3])


    # set up the spark context
    homedir = os.environ['HOME']
    os.environ['SPARK_HOME'] = '%s/spark'%homedir
    spark_home = os.environ['SPARK_HOME']

    os.system('~/spark/sbin/start-all.sh')

    master_url = 'spark://%s:7077'%os.environ['HOST']
    sys.path.insert(0,os.environ['SPARK_HOME']+'/python')
    sys.path.insert(0,os.environ['SPARK_HOME']+'/python/lib/py4j-0.8.1-src.zip')

    import pyspark
    from pyspark import SparkContext

    sc = SparkContext(master = master_url, appName = 'sparkgram unit tests', batchSize=10)

    dv = SparkDocumentVectorizer(sc, short_doclist, ngram_range = [1,3],
                                 tokenizer = sparkgram.document_vectorizer.alpha_tokenizer)


def test_feature_count() :
    svs = dv.docvec_rdd.values().collect()
    cv_mat = cv.fit_transform(short_doclist)

    for i,sv in enumerate(svs):
        assert(len(sv.values) == cv_mat.getrow(i).getnnz())


def test_feature_names():
    cv.fit(short_doclist)
    cv_vocab = cv.get_feature_names()
    dv_vocab = dv.vocab_rdd.collect()
    assert(cv_vocab == dv_vocab)


def test_vocab_hash_collisions_short() :
    nunique = len(np.unique(dv.get_vocab_map().values().collect()))
    nterms = len(dv.vocab_rdd.collect())
    assert(nunique == nterms)


def teardown() :
    sc.stop()
    os.system('~/spark/sbin/stop-all.sh')
    os.system('rm test_vectors*.npz')


if __name__ == '__main__' :
    setup()
    test_feature_count()
    test_feature_names()
    test_vocab_hash_collisions()
    #teardown()
