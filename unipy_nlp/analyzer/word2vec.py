# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:41:19 2019

@author: Youngju Jaden Kim
"""

import os
import re
import sys
import json
import random
import warnings
import subprocess
import itertools as it
import functools as ft
import collections
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd

import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import unidecode
from unicodedata import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


__all__ = []
__all__ += [
    'Word2Vec',
]

class Word2Vec(object):

    def __init__(
            self,
            tokenized_sentence_list,
            ):

        self.trained = False
        self.tokenized_sentence_list = tokenized_sentence_list


    def train_w2v(
            self,
            size=70,
            window=4,
            min_count=10,
            negative=16,
            workers=8,
            iter=50,
            sg=1,
            ):

        self.embed = gensim.models.Word2Vec(
            self.tokenized_sentence_list,
            size=size,
            window=window,
            min_count=min_count,
            negative=negative,
            workers=workers,
            iter=iter,
            sg=sg,
        )
        self.trained = True

    def save_w2v(self, filepath):
        dirpath, filename = os.path.split(filepath)
        os.makedirs(dirpath, exist_ok=True)
        self.embed.save(filepath)
        self.trained = True

    def load_w2v(self, filepath):
        self.embed = gensim.models.Word2Vec.load(filepath)
        self.trained = True

    def get_similar(self, words, top_n=2):

        if not self.trained:
            raise Exception('Train `Word2Vec` first.')

        if isinstance(words, str):
            words = [words]

        try:
            res = self.embed.wv.most_similar(
                positive=words,
                topn=top_n,
            )
            return res
        except KeyError as err:
            print(err)
            return None

    def save_tensorboard(self, dirpath=None):

        if not self.trained:
            raise Exception('Train `Word2Vec` first.')

        os.makedirs(dirpath, exist_ok=True)

        weights = self.embed.wv.vectors
        idx2words = self.embed.wv.index2word

        vocab_size = weights.shape[0]
        embedding_dim = weights.shape[1]

        with open(os.path.join(dirpath, "metadata.tsv"), 'w') as f:
            f.writelines("\n".join(idx2words))

        tf.reset_default_graph()

        W = tf.Variable(
            tf.constant(0., shape=[vocab_size, embedding_dim]),
            trainable=False,
            name="W",
        )
        embedding_placeholder = tf.placeholder(
            tf.float32,
            [vocab_size, embedding_dim],
        )
        embedding_init = W.assign(embedding_placeholder)

        writer = tf.summary.FileWriter(
            dirpath,
            graph=tf.get_default_graph(),
        )
        saver = tf.train.Saver()

        # tf.contrib.tensorboard.plugins.projector.projector_config.proto
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = W.name
        embedding.metadata_path = os.path.join(
            # filepath,
            "metadata.tsv",
        )
        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(writer, config)

        with tf.Session() as sess:
            sess.run(
                embedding_init,
                feed_dict={embedding_placeholder: weights},
            )
            save_path = saver.save(
                sess,
                os.path.join(
                    dirpath,
                    "tf-model.cpkt",
                ),
            )
        
        print(f"'Projector Saved: '{save_path}'")

        return save_path
