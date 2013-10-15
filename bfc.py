#!/usr/bin/env python

import argparse
import logging

from itertools import izip_longest

import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import v_measure_score

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
    parser.add_argument('--tax-level', default='Phylum', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--n-samples', default=1000)

    return parser.parse_args()


def grouper(iterable, n, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def iter_chunk(records, size, level):

    iter_list = []

    for i in range(0, len(records), size):

        logging.debug('sampling the %s -th through %s -th sample' %(i, i+size))

        chunk = records[i:i+size]

        logging.debug('got chunk of size %s' % len(chunk))

        # throw out records with ambiguous nucleotides
        labels = [ get_label(r, level) for r in records ]
        features = [ str(r.seq) for r in records ]

        iter_list.append((labels, features))

    logging.info('chunked %s chunks' % len(iter_list))
    return iter_list


def get_label(record, level):
    ''' For now, just returns Phylum '''
    levels = {
            'Domain': 0,
            'Phylum': 1,
            'Class': 2,
            'Order': 3,
            'Family': 4,
            'Genus': 5,
            'Species': 6
            }

    level = levels[level]

    return record.description.split(';')[level]


def get_classes(records, level):
    classes = list(set( get_label(i, level) for i in records ))

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

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(filename='/dev/stderr', level=log_level)

    hasher = HashingVectorizer(analyzer='char',
                               n_features = 2 ** 18,
                               ngram_range=(args.ngram_min, args.ngram_max))

    logging.info(hasher)

    classifier = MiniBatchKMeans()

    with open(args.fasta_file) as handle:

        logging.info('using taxonomy level %s' % args.tax_level)
        logging.info('ngram range: [%s-%s]' % (args.ngram_min, args.ngram_max))

        logging.info('loading data')
        records = list(SeqIO.parse(handle, 'fasta'))

        encoder, classes = get_classes(records, args.tax_level)

        records = records[0:args.n_samples]

        chunk_generator = iter_chunk(records, args.chunk_size, args.tax_level)

        for labels, features in chunk_generator:

            logging.info('transforming training chunk')
            labels = encoder.transform(labels)
            vectors = hasher.transform(features)

            logging.info('fitting training chunk')
            classifier.partial_fit(vectors)

            pred_labels = classifier.predict(vectors)

            score = v_measure_score(labels, pred_labels)
            shuffled_score = v_measure_score(labels, sample(pred_labels, len(pred_labels)))

            logging.info('score: %.2f' % (score))
            logging.info('shuffled score: %.2f' % (shuffled_score))


if __name__ == '__main__':
    main()
