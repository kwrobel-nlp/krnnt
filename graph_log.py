#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':
    parser = ArgumentParser(description='Plots data for graph')
    parser.add_argument('output_path', help='output path to features dict')
    parser.add_argument('input_path', help='input path to log data')


    args = parser.parse_args()


    test_scores=[]
    dev_scores=[]
    with open(args.input_path) as file:
        for line in file:
            m = re.search(r'\'val_score\', (.*?)\)', line)
            if m is None:
                continue
            test_score=float(m.group(1))

            m = re.search(r'\'dev_val_score\', (.*?)\)', line)
            if m is None:
                continue
            dev_scores += (float(m.group(1)),)
            test_scores+=(test_score, )

    t=range(len(test_scores))
    plt.plot(test_scores)

    if any([score!=0.0 for score in dev_scores]):
        plt.plot(dev_scores)
    plt.ylabel('some numbers')
    plt.show()

    print('Test scores:')
    for score in test_scores:
        print(score)

    print('Dev scores:')
    for score in dev_scores:
        print(score)