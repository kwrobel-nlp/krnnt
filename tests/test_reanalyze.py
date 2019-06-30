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


def test_maca():
    paragraph_raw = 'Lubię pociągi.'
    try:
        results = Preprocess.maca([paragraph_raw], maca_config='morfeusz-nkjp-official')
        results = list(results)
    except:
        results = Preprocess.maca([paragraph_raw], maca_config='morfeusz2-nkjp')
        results = list(results)
    print(results[0])
    assert len(results) == 1
    assert results[0] == reference_maca_output


def test_parse():
    result = Preprocess.parse(reference_maca_output)

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