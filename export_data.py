from krnnt.serial_pickle import SerialUnpickler
from tqdm import tqdm

from krnnt.structure import Paragraph


#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from krnnt.serial_pickle import SerialUnpickler
from krnnt.writers import get_output_converter


def paragraph_to_result(paragraph: Paragraph):

    paragraph2=[]
    for sentence in paragraph:
        try:
            sentence2=[]
            paragraph2.append(sentence2)
            for token in sentence:
                sentence2.append({
                    'token':token.form,
                    'sep':token.space_before,
                    'tag': token.gold_form.tags,
                    'lemmas': [token.gold_form.lemma],
                })
        except AttributeError: #omit sentence if some token does no have gold tag
            continue
    return paragraph2

if __name__ == '__main__':
    parser = ArgumentParser(description='Export data (before preprocessing) to format')
    parser.add_argument('input_path', help='input path to data')
    parser.add_argument('output_path', help='output path to data')
    parser.add_argument('-f','--format', default='txt', help='output format')

    args = parser.parse_args()

    with open(args.input_path, 'rb') as file:
        su = SerialUnpickler(file)

        converter=get_output_converter(args.format)

        string=converter((paragraph_to_result(paragraph_gold) for paragraph_gold in su))

        with open(args.output_path, 'w') as output_file:
            output_file.write(string)


