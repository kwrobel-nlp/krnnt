#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Join features dicts')
    parser.add_argument('output_path', help='output path to features dict')
    parser.add_argument('input_paths', nargs='+', help='input paths to features dicts')

    parser.add_argument('--reproducible', action='store_true', default=False, help='set seeds')

    args = parser.parse_args()

    if args.reproducible:
        from numpy.random import seed

        seed(1337)
        import random as rn

        rn.seed(1337)

    print(args.input_paths)
    joined_unique_features_dict = None
    for input_path in args.input_paths:
        unique_features_dict = pickle.load(open(input_path, 'rb'))

        if joined_unique_features_dict is None:
            joined_unique_features_dict = unique_features_dict
        else:
            for feature_name, dict2 in unique_features_dict.items():

                if feature_name not in joined_unique_features_dict:
                    joined_index = 0
                else:
                    joined_index = max(joined_unique_features_dict[feature_name].values()) + 1
                    assert joined_index == len(joined_unique_features_dict[feature_name])

                for value, index in sorted(dict2.items(), key=lambda x: x[1]):
                    if value not in joined_unique_features_dict[feature_name]:
                        joined_unique_features_dict[feature_name][value] = joined_index
                        joined_index += 1

    with open(args.output_path, 'wb') as file:
        pickle.dump(joined_unique_features_dict, file)
