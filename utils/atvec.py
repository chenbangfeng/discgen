import argparse
import sys
import json
import numpy as np
from utils.fuel_helper import get_dataset_iterator
from plat.utils import json_list_to_array

def get_averages(attribs, encoded, num_encoded_attributes):
    with_attr = [[] for x in xrange(num_encoded_attributes)]
    without_attr = [[] for x in xrange(num_encoded_attributes)]
    for i in range(len(encoded)):
        if i % 10000 == 0:
            print("iteration {}".format(i))
        for m in range(num_encoded_attributes):
            if attribs[i][0][m] == 1:
                with_attr[m].append(encoded[i])
            else:
                without_attr[m].append(encoded[i])

    print("With: {}".format(map(len, with_attr)))
    print("Without: {}".format(map(len, without_attr)))

    with_attr = map(np.array, with_attr)
    without_attr = map(np.array, without_attr)
    return with_attr, without_attr

def averages_to_attribute_vectors(with_attr, without_attr, num_encoded_attributes, latent_dim):
    atvecs = np.zeros((num_encoded_attributes, latent_dim))
    for n in range(num_encoded_attributes):
        m1 = np.mean(with_attr[n],axis=0)
        m2 = np.mean(without_attr[n],axis=0)
        atvecs[n] = m1 - m2
    return atvecs

def save_json_attribs(attribs, filename):
    with open(filename, 'w') as outfile:
        json.dump(attribs.tolist(), outfile)   

def main(cliargs):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset (for labels).")
    parser.add_argument('--split', dest='split', default="train",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--num-attribs", dest='num_attribs', type=int, default=40,
                        help="Number of attributes (labes)")
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension of vectors")
    parser.add_argument("--encoded-vectors", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument('--outfile', dest='outfile', default=None,
                        help="Output json file for vectors.")
    args = parser.parse_args(cliargs)

    encoded = json_list_to_array(args.encoded_vectors)
    num_rows, z_dim = encoded.shape
    attribs = np.array(list(get_dataset_iterator(args.dataset, args.split, include_features=False, include_targets=True)))
    print("encoded vectors: {}, attributes: {} ".format(encoded.shape, attribs.shape))

    with_attr, without_attr = get_averages(attribs, encoded, args.num_attribs);
    atvects = averages_to_attribute_vectors(with_attr, without_attr, args.num_attribs, z_dim)
    print("Computed atvecs shape: {}".format(atvects.shape))

    if args.outfile is not None:
        save_json_attribs(atvects, args.outfile)

if __name__ == '__main__':
    main(sys.argv[1:])
