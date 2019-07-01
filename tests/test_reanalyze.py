from krnnt.analyzers import MacaAnalyzer
from krnnt.classes import Form
from krnnt.pipeline import Preprocess

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
        results = maca_analyzer._maca([paragraph_raw])
        results = list(results)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        results = maca_analyzer._maca([paragraph_raw])
        results = list(results)

    print(results[0])
    assert len(results) == 1
    assert results[0] == reference_maca_output


def test_parse():
    maca_analyzer = MacaAnalyzer('')
    result = maca_analyzer._parse(reference_maca_output)

    reference = [
        ('Lubię', 'newline',
         [('lubić', 'fin:sg:pri:imperf')]),
        ('pociągi', 'space',
         [('pociąg', 'subst:pl:nom:m3'),
          ('pociąg', 'subst:pl:acc:m3'),
          ('pociąg', 'subst:pl:voc:m3')]),
        ('.', 'none',
         [('.', 'interp')])]

    assert result == reference

def test_maca_analyzer():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        result = maca_analyzer.analyze(paragraph_raw)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        result = maca_analyzer.analyze(paragraph_raw)

    assert len(result.sentences)==1
    assert len(result.sentences[0].tokens) == 3

    assert result.sentences[0].tokens[0].form == 'Lubię'
    assert result.sentences[0].tokens[0].space_before == True
    assert len(result.sentences[0].tokens[0].interpretations) == 1

    assert result.sentences[0].tokens[1].form == 'pociągi'
    assert result.sentences[0].tokens[1].space_before == True
    assert len(result.sentences[0].tokens[1].interpretations) == 3

    assert result.sentences[0].tokens[2].form == '.'
    assert result.sentences[0].tokens[2].space_before == False
    assert len(result.sentences[0].tokens[2].interpretations) == 1

    assert result.sentences[0].tokens[1].interpretations[0] == Form('pociąg', 'subst:pl:nom:m3')
    assert result.sentences[0].tokens[1].interpretations[1] == Form('pociąg', 'subst:pl:acc:m3')
    assert result.sentences[0].tokens[1].interpretations[2] == Form('pociąg', 'subst:pl:voc:m3')
