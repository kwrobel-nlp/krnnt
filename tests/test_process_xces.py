import os

from krnnt.aligner import align_paragraphs
from krnnt.analyzers import MacaAnalyzer
from krnnt.readers import read_xces

#TODO parametrize?



def test_different_xces_formats(rootdir):
    data = {
        os.path.join(rootdir, 'data/small/nkjp1m-1.2-xces.xml'): [8, 7],
        os.path.join(rootdir, 'data/small/train-gold.xml'): [10, 8, 6],
        os.path.join(rootdir, 'data/small/gold-task-c.xml'): [12, 12],
        os.path.join(rootdir, 'data/small/00130846.ann.xml'): [25],
        os.path.join(rootdir, 'data/small/00130846.xml'): [25],
        os.path.join(rootdir, 'data/small/00132482.ann.xml'): [2],
        os.path.join(rootdir, 'data/small/00132482.xml'): [2]
    }

    for path, paragraph_lenghts in data.items():
        assert paragraph_lenghts == [len(paragraph.sentences) for paragraph in read_xces(path)]
        for paragraph in read_xces(path):
            print(paragraph.text())

            for sentence in paragraph:
                for token in sentence:
                    print(token)
        print()

def test_reanalyze(rootdir):
    data = {
        os.path.join(rootdir, 'data/small/nkjp1m-1.2-xces.xml'): [8, 7],
        os.path.join(rootdir, 'data/small/train-gold.xml'): [10, 8, 6],
        os.path.join(rootdir, 'data/small/gold-task-c.xml'): [12, 12],
        os.path.join(rootdir, 'data/small/00130846.ann.xml'): [25],
        os.path.join(rootdir, 'data/small/00130846.xml'): [25],
        os.path.join(rootdir, 'data/small/00132482.ann.xml'): [2],
        os.path.join(rootdir, 'data/small/00132482.xml'): [2]
    }

    for path, paragraph_lenghts in data.items():
        # assert paragraph_lenghts == [len(paragraph.sentences) for paragraph in read_xces(path)]
        maca_analyzer = MacaAnalyzer('morfeusz2-nkjp')
        for paragraph in read_xces(path):
            paragraph_raw = paragraph.text()

            paragraph_reanalyzed = maca_analyzer.analyze(paragraph_raw)

            print('Number of sentences by Maca vs gold', len(paragraph_reanalyzed.sentences), len(paragraph.sentences))

            paragraph_reanalyzed = align_paragraphs(paragraph_reanalyzed, paragraph)
            for sentence in paragraph_reanalyzed:
                for token in sentence:
                    print(token)
        print()
