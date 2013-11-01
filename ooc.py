#!/usr/bin/env python

# One-off Python script boiler plate v0.0.1
# (c) @heyaudy, 2013. License = MITv3

# on-line, unsupervised clustering.

import argparse
import logging

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from Bio import SeqIO


def parse_args():
    ''' return arguments
        >>> args = parse_args()
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--log', default='/dev/stderr', help='log file (default=stderr)')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--fasta-file', default='/dev/stdin')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--ngram-min', default=1, type=int)
    parser.add_argument('--ngram-max', default=1, type=int)

    return parser.parse_args()


def sequence_chunk_generator(fasta_file, chunk_size=0):
    ''' '''

    records = SeqIO.parse(fasta_file, 'fasta')

    chunk = []

    logging.info('chunking records into chunks of size %s' % chunk_size)

    for record in records:

        chunk.append(record)

        logging.debug('chunk size: %s' % len(chunk))

        if len(chunk) > chunk_size:
            logging.debug('yielding chunk')
            yield chunk
            logging.debug('resetting chunk')
            chunk = []


def setup_logging(logfile, verbose=False):
    ''' sets up logging
        >>> setup_logging('logfile.txt', verbose=True)
    '''

    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(filename=logfile, level=log_level)



def main():
    '''
        >>> main() # stuff happens
    '''

    args = parse_args()
    setup_logging(args.log, verbose=args.verbose)

    chunks = sequence_chunk_generator(args.fasta_file,
                                      chunk_size=args.chunk_size)

    hasher = HashingVectorizer(analyzer='char',
                               n_features = 2 ** 18,
                               ngram_range=(args.ngram_min, args.ngram_max),
                               )

    estimator = AffinityPropagation()

    for chunk in chunks:

        logging.info('hashing chunk')
        chunk_vector = hasher.transform([ str(i.seq) for i in chunk ])

        logging.info('clustering')

        estimator.fit(chunk_vector)

        logging.info('got %s clusters' % len(set(estimator.labels_)))



if __name__ == '__main__':
    main()
