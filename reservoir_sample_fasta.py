#!/usr/bin/env python

# Perform reservoir sampling of records in a FASTA file
# (c) @heyaudy, 2013. License = MITv3

import argparse
import logging
import random
from Bio import SeqIO

def parse_args():
    ''' returns command-line arguments parsed by argparse.
        >>> args = parse_args()
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='/dev/stderr', help='log file (default=stderr)')
    parser.add_argument('--fasta-file')
    parser.add_argument('--n-sequences', type=int)
    parser.add_argument('--output-fasta', default='/dev/stdout')
    return parser.parse_args()

def reservoir_sample_fasta(records, n_sequences):
    ''' Returns random sample of SeqIO records.

        >>> with open('very_large.fasta') as handle:
        ...     records = SeqIO.parse(handle, 'fasta')
        ...     one_hundred_records = reservoir_sample_fasta(handle, 100)

        >>> print len(one_hundred_records)
        100

    '''

    sample = []

    for i, record in enumerate(records, start = 1):

        if i < n_sequences:
            sample.append(record)
            logging.debug('current reservoir size: %s' % len(sample))

        elif i >= n_sequences and random.random() < n_sequences/float(i):
            replace = random.randint(0, len(sample) - 1)
            sample[replace] = record
            logging.debug('reservoir full')

    return sample


def main():
    ''' main function.
        >>> main() # stuff happens
    '''

    args = parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO)

    logging.info('sampling %s sequences from %s' % (args.n_sequences, args.fasta_file))

    with open(args.fasta_file) as handle:
        records = SeqIO.parse(handle, 'fasta')
        sample = reservoir_sample_fasta(records, args.n_sequences)
        logging.info('sampled %s sequences', len(sample))

    logging.info('writing sequences to %s' % args.output_fasta)
    with open(args.output_fasta, 'w') as handle:
        SeqIO.write(sample, handle, 'fasta')


if __name__ == '__main__':
    main()
