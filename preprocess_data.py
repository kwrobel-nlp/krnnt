from argparse import ArgumentParser

from tqdm import tqdm

from krnnt.new import preprocess_paragraph_preanalyzed, \
    preprocess_paragraph_reanalyzed
from krnnt.serial_pickle import SerialPickler, SerialUnpickler
from krnnt.structure import Paragraph

if __name__ == '__main__':
    parser = ArgumentParser(description='Create features for neural network.')
    parser.add_argument('input_path', type=str, help='path to re/preanalyzed data')
    parser.add_argument('output_path', type=str, help='save path')
    parser.add_argument('-p', '--preanalyzed', action='store_false',
                        default=True, dest='reanalyzed',
                        help='training data have not been reanalyzed')
    args = parser.parse_args()

    file = open(args.input_path, 'rb')
    su = SerialUnpickler(file)

    file2 = open(args.output_path, 'wb')
    sp = SerialPickler(file2)

    if args.reanalyzed:
        preprocess_method = preprocess_paragraph_reanalyzed
    else:
        preprocess_method = preprocess_paragraph_preanalyzed

    paragraph: Paragraph
    for paragraph in tqdm(su, total=18484):
        paragraph_sequence = preprocess_method(paragraph)

        sp.add(paragraph_sequence)

    file.close()
    file2.close()

