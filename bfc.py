#!/usr/bin/env python

import argparse
import logging

from itertools import izip_longest

import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import cross_val_score

from Bio import SeqIO

from random import sample


ALLOWED_CHARS = [ 'G', 'A', 'T', 'C' ]

def parse_args():
    ''' doctstring for parse_args '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta-file')
    parser.add_argument('--chunk-size', default=10, type=int)
    parser.add_argument('--ngram-min', default=10, type=int)
    parser.add_argument('--ngram-max', default=12, type=int)
    return parser.parse_args()


def grouper(iterable, n, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def iter_chunk(records, size):

    while True:

        logging.info('getting chunk of size %s' % size)

        chunk = sample(records, size)

        # throw out records with ambiguous nucleotides
        labels = [ get_label(r, 'Phylum') for r in records ]
        features = [ str(r.seq) for r in records ]

        yield labels, features


def get_label(record, level):
    ''' For now, just returns Phylum '''
    return record.description.split(';')[1].lstrip('[1]')


def get_classes(records):
    classes = list(set( get_label(i, 'Phylum') for i in records ))

    encoder = LabelEncoder()
    encoder.fit_transform(classes)

    classes = np.unique(classes)

    return encoder, classes

def dna_to_binary(s):
    binary_table = {
            'G': 0b00,
            'A': 0b01,
            'T': 0b10,
            'C': 0b11
            }
    return [ binary_table[i.upper()] for i in s ]


def main():
    ''' doctsring for main '''

    args = parse_args()

    logging.basicConfig(filename='/dev/stderr', level=logging.DEBUG)

    hasher = HashingVectorizer(analyzer='char',
                               n_features = 2 ** 18,
                               ngram_range=(args.ngram_min, args.ngram_max))

    logging.info(hasher)

    classifier = PassiveAggressiveClassifier()
    dummy_classifier = DummyClassifier()

    with open(args.fasta_file) as handle:
        logging.info('loading data')
        records = list(SeqIO.parse(handle, 'fasta'))

        encoder, classes = get_classes(records)

        chunk_generator = iter_chunk(records, args.chunk_size)

        logging.info('fetching testing chunk')
        t_labels, t_features = chunk_generator.next()

        logging.info('transforming testing chunk')
        t_labels = encoder.transform(t_labels)
        t_features = hasher.transform(t_features)

        for labels, features in chunk_generator:

            logging.info('transforming training chunk')
            labels = encoder.transform(labels)
            vectors = hasher.transform(features)

            logging.info('fitting training chunk')
            classifier.partial_fit(vectors, labels, classes=classes)

            logging.info('cross-validating w/ testing chunk')
            scores = cross_val_score(classifier, t_features, t_labels)
            dummy_scores = cross_val_score(dummy_classifier, t_features, t_labels)

            logging.info('score: %.2f (SD=%.2f)' % (np.mean(scores), np.std(scores)))
            logging.info('dummy score: %.2f (SD=%.2f)' % (np.mean(dummy_scores), np.std(dummy_scores)))




if __name__ == '__main__':
    main()
