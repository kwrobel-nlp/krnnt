import os

from krnnt.analyzers import MacaAnalyzer
from krnnt.new import get_morfeusz, analyze_tokenized, analyze_token
from krnnt.structure import Form

reference_maca_output = \
'''Lubię	newline
	lubić	fin:sg:pri:imperf
pociągi	space
	pociąg	subst:pl:nom:m3
	pociąg	subst:pl:acc:m3
	pociąg	subst:pl:voc:m3
.	none
	.	interp'''

paragraph_raw = 'Lubię pociągi.'

MACA_CONFIG1='morfeusz-nkjp-official'
MACA_CONFIG2='morfeusz2-nkjp'

def test_maca():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        results = maca_analyzer._maca(paragraph_raw)
        results = list(results)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        results = maca_analyzer._maca(paragraph_raw)
        results = list(results)

    assert len(results) == 1
    assert results[0] == reference_maca_output


def test_maca_analyzer(rootdir):
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        result = maca_analyzer.analyze(paragraph_raw)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        result = maca_analyzer.analyze(paragraph_raw)

    lines = []
    for line in open(os.path.join(rootdir, 'data/full/test-raw.txt')):
        line = line.strip()
        if not line: continue
        lines.append(line)

    morfeusz = get_morfeusz()



    for line in lines:
        paragraph = maca_analyzer.analyze(line)
        for sentence in paragraph:
            for token in sentence:

                maca_tags=[(form.lemma, form.tags) for form in token.interpretations]
                morfeusz_tags=analyze_token(morfeusz, token.form)
                maca_tags=set(maca_tags)
                morfeusz_tags=set(morfeusz_tags)
                if maca_tags!=morfeusz_tags:
                    print(token)
                    print(sorted(maca_tags-morfeusz_tags))
                    print(sorted(morfeusz_tags-maca_tags))