from krnnt.analyzers import MacaAnalyzer
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
        results = maca_analyzer._maca([paragraph_raw])
        results = list(results)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        results = maca_analyzer._maca([paragraph_raw])
        results = list(results)

    assert len(results) == 1
    assert results[0] == reference_maca_output

def test_maca_process():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        results = maca_analyzer._maca_process([paragraph_raw])
        results = list(results)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        results = maca_analyzer._maca_process([paragraph_raw])
        results = list(results)

    assert len(results) == 1
    assert results[0] == reference_maca_output

def test_maca_wrapper():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        results = maca_analyzer._maca_wrapper([paragraph_raw])
        results = list(results)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        results = maca_analyzer._maca_wrapper([paragraph_raw])
        results = list(results)

    assert len(results) == 1
    assert results[0] == reference_maca_output

def test_parse():
    maca_analyzer = MacaAnalyzer('')
    maca_analyzer.text = paragraph_raw
    maca_analyzer.last_offset = 0
    result = maca_analyzer._parse(reference_maca_output)

    reference = [
        ('Lubię', 'newline',
         [('lubić', 'fin:sg:pri:imperf')],0,5),
        ('pociągi', 'space',
         [('pociąg', 'subst:pl:nom:m3'),
          ('pociąg', 'subst:pl:acc:m3'),
          ('pociąg', 'subst:pl:voc:m3')],6,13),
        ('.', 'none',
         [('.', 'interp')], 13,14)]

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
    assert result.sentences[0].tokens[0].space_before == 'newline'
    assert len(result.sentences[0].tokens[0].interpretations) == 1

    assert result.sentences[0].tokens[1].form == 'pociągi'
    assert result.sentences[0].tokens[1].space_before == 'space'
    assert len(result.sentences[0].tokens[1].interpretations) == 3

    assert result.sentences[0].tokens[2].form == '.'
    assert result.sentences[0].tokens[2].space_before == 'none'
    assert len(result.sentences[0].tokens[2].interpretations) == 1

    assert result.sentences[0].tokens[1].interpretations[0] == Form('pociąg', 'subst:pl:nom:m3')
    assert result.sentences[0].tokens[1].interpretations[1] == Form('pociąg', 'subst:pl:acc:m3')
    assert result.sentences[0].tokens[1].interpretations[2] == Form('pociąg', 'subst:pl:voc:m3')


def test_maca_analyzer_lemmas():
    paragraph_raw='Ala ma kota.'
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        result = maca_analyzer.analyze(paragraph_raw)
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        result = maca_analyzer.analyze(paragraph_raw)

    lemmas =[form.lemma for form in result.sentences[0].tokens[2].interpretations]
    assert 'kot' in lemmas
    assert 'kot:s1' not in lemmas
    assert 'kot:s2' not in lemmas

