#!/usr/bin/env python

import os
import argparse
import logging
from itertools import izip_longest

import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from Bio import SeqIO

def parse_args():
    ''' doctstring for parse_args '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--fasta-file')
    parser.add_argument('--ngram-min', default=10, type=int)
    parser.add_argument('--ngram-max', default=12, type=int)
    parser.add_argument('--n-clusters', default=2, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--n-iters', default=10, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--not-paired', default=True, action='store_false')
    parser.add_argument('--tfid', default=False, action='store_true')
    parser.add_argument('--log-file', default='/dev/stderr')
    parser.add_argument('--seqs-out', default=False, help='file name for labelled sequences')
    parser.add_argument('--benchmark', default=False, help='benchmark (taxonomic level)')

    return parser.parse_args()


def setup_logging(logfile='/dev/stderr', verbose=False):
    ''' sets up the logger based on the verbosity '''

    # Setup logging
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(filename=logfile, level=log_level)


def fasta_pair_iterator(fasta_file, paired=True):
    ''' Consumes an entire FASTA file returning
    a list of tuples of SeqIO record pairs. '''

    logging.info('consuming %s' % fasta_file)

    with open(fasta_file) as handle:
        records = list(SeqIO.parse(handle, 'fasta'))

    def grouper(n, iterable, fillvalue=None):
        "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return izip_longest(fillvalue=fillvalue, *args)

    if paired:
        pairs = list(grouper(2, records))
    else:
        pairs = list(grouper(1, records))

    return pairs


def main():
    ''' doctsring for main '''

    args = parse_args()

    setup_logging(logfile=args.log_file, verbose = args.verbose)

    pairs = fasta_pair_iterator(args.fasta_file)

    pairs = [ p for p in pairs if not None in p ]

    features = [ "%s %s" % (p[0].seq, p[1].seq) for p in pairs ]

    # setup Hasher, Vectorizer and Classifier
    hasher = HashingVectorizer(analyzer='char_wb',
                               n_features = 2 ** 18,
                               ngram_range=(args.ngram_min, args.ngram_max),
                               )
    if args.tfid:
        logging.info('Using Tfidf Normalization')
        normalizer = TfidfTransformer(use_idf=True)
        hasher = Pipeline([('hash', hasher),
                           ('normalize', normalizer)])

    logging.info(hasher)

    logging.info('Using %s clusters' % args.n_clusters)

    estimator = MiniBatchKMeans(n_clusters = args.n_clusters)

    logging.info('Estimator: %s' % estimator)

    logging.info('ngram range: [%s-%s]' % (args.ngram_min, args.ngram_max))

    logging.info('hashing features')
    vectors = hasher.fit_transform(features)

    dimx = vectors.shape[0]
    for i in range(0, args.n_iters):
        logging.info('iterating %s/%s' % (i, args.n_iters))

        choice_matrix = np.random.randint(dimx, size=args.batch_size)

        subset = vectors[choice_matrix,:]

        logging.info('fitting subset')

        estimator.partial_fit(subset)

    logging.info('predicting all vectors')
    clusters = estimator.predict(vectors)

    logging.info('dimension reduction using LDA')
    lda = TruncatedSVD(n_components=3)

    coords = lda.fit_transform(vectors)
    coords = Normalizer(copy=False).fit_transform(coords)

    logging.info('plotting')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    p = ax.scatter(x, y, z, c=clusters)

    fig.colorbar(p)

    plt.axis('tight')
    plt.savefig('nmds.png')

    # get cluster centroids
    # these are the centroids in kmer-spectrum space...
    # need to find a way to get back original sequences
    # (that created the centroids... ?)
    centroids = estimator.cluster_centers_

    # get sequences for each cluster
    if args.seqs_out:
        with open(args.seqs_out, 'w') as handle:
            for c, f in zip(clusters, pairs):
                for ff in f:
                    print >> handle, '>cluster=%s %s\n%s' % (c, ff.description, ff.seq)



if __name__ == '__main__':
    estimator = main()
