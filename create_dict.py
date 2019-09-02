import json
import pickle
from argparse import ArgumentParser

from tqdm import tqdm
import jsonlines

from krnnt.new import preprocess_paragraph_preanalyzed, \
    preprocess_paragraph_reanalyzed, serialize_sample_paragraph, create_dict
from krnnt.serial_pickle import SerialPickler, SerialUnpickler
from krnnt.structure import Paragraph

if __name__ == '__main__':
    parser = ArgumentParser(description='Create dictionary of features')
    parser.add_argument('input_path', type=str, help='path to preprocessed data')
    parser.add_argument('output_path', type=str, help='save path')
    args = parser.parse_args()

    file = open(args.input_path, 'rb')
    su = SerialUnpickler(file)

    unique_dict = create_dict(su)

    with open(args.output_path, 'wb') as file:
        pickle.dump(unique_dict, file)

    with open(args.output_path+'.json','w') as file:
        json.dump(unique_dict, file, ensure_ascii=False)