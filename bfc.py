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
    parser.add_argument('--n-iters', default=1000)

    return parser.parse_args()


def grouper(iterable, n, fillvalue=None):
    '''
    Chunks an iterable into n-sized chunks. Optionally, specify
    fillvalue to fill overflows
    '''

    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def iter_chunk(records, size, level):
    ''' Iterates over records return chunks of a specified size '''

    iter_list = []

    for i in range(0, len(records), size):

        logging.debug('sampling the %s -th through %s -th sample' %(i, i+size))

        chunk = records[i:i+size]

        logging.debug('got chunk of size %s' % len(chunk))

        # throw out records with ambiguous nucleotides
        labels = [ get_label(r.description, level) for r in records ]
        features = [ str(r.seq) for r in records ]

        iter_list.append((labels, features))

    logging.info('chunked %s chunks' % len(iter_list))
    return iter_list


def get_label(taxonomy, level):
    ''' Return the label given a TaxCollector string and
        taxonomic level i.e. Genus

        >>> label = get_label('[0]Domain[1]Phylum[2]Class[3]Order[4]Family[5]Genus[6]Species', 'Species')
        >>> print label
        Species

    '''

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

    return taxonomy.split(';')[level]


def get_classes(records, level):
    ''' Returns list of classes given a list of SeqIO records.
        Classes are defined by taxonomic descriptions at a specific
        level.

        For example, the classes for a list of taxonomic descriptions at the
        Phylum level would be a list of all the phyla included in the list of
        records
    '''

    classes = list(set( get_label(i.description, level) for i in records ))

    encoder = LabelEncoder()
    encoder.fit_transform(classes)

    classes = np.unique(classes)

    return encoder, classes


def dna_to_binary(s):
    ''' Convert a string of nucleotides {G, A, T, C} to a 2-bit
    binary representation.
    '''

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

    logging.info('consuming %s' % args.fasta_file)

    with open(args.fasta_file) as handle:
        records = list(SeqIO.parse(handle, 'fasta'))

    logging.info('consumed %s records' % len(records))

    encoder, classes = get_classes(records, args.tax_level)
    n_clusters = len(classes)

    logging.info('using taxonomic level %s' % args.tax_level)
    logging.info('Using %s clusters' % n_clusters)

    classifier = MiniBatchKMeans(n_clusters = n_clusters)

    records = records[0:args.n_iters]

    chunk_generator = iter_chunk(records, args.chunk_size, args.tax_level)

    logging.info('ngram range: [%s-%s]' % (args.ngram_min, args.ngram_max))

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
