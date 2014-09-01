#!/usr/bin/env python

from distutils.core import setup

setup(name='spark_ngram',
      description='Simple ngram tools to use with Spark',
      author='Rok Roskar',
      author_email='rokroskar@gmail.com',
      url='http://github.com/rokroskar/spark_ngram',
      packages=['spark_ngram', 'spark_ngram/vectorizers'],
     )
