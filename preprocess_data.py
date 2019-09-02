from argparse import ArgumentParser

from tqdm import tqdm
import jsonlines

from krnnt.new import preprocess_paragraph_preanalyzed, \
    preprocess_paragraph_reanalyzed, serialize_sample_paragraph
from krnnt.serial_pickle import SerialPickler, SerialUnpickler
from krnnt.structure import Paragraph

if __name__ == '__main__':
    parser = ArgumentParser(description='Create features for neural network')
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

    jf = jsonlines.open(args.output_path + '.jsonl', mode='w')

    if args.reanalyzed:
        preprocess_method = preprocess_paragraph_reanalyzed
    else:
        preprocess_method = preprocess_paragraph_preanalyzed

    paragraph: Paragraph
    for paragraph in tqdm(su, total=18484, desc='Processing'):
        paragraph_sequence = preprocess_method(paragraph)

        jf.write(serialize_sample_paragraph(paragraph_sequence))
        sp.add(paragraph_sequence)

    file.close()
    file2.close()
