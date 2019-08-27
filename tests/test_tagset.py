import pytest
from krnnt.analyzers import MacaAnalyzer
from krnnt.new import get_morfeusz, analyze_token

MACA_CONFIG1='morfeusz-nkjp-official'
MACA_CONFIG2='morfeusz2-nkjp'

@pytest.fixture
def maca():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        list(maca_analyzer._maca("test"))
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        list(maca_analyzer._maca("test"))

    return maca_analyzer

test_data = [
    ('IV', '', 'num:::'),
    ('IV', '', 'romandig'),
    ('1', '', 'dig'),
    ('prostu', 'adjp', 'adjp:gen'),
    (':)', '', 'emo'),
    ('godzien', 'adjc', ''),
    ('oślep', 'burk', 'frag'),
    ('obojga', 'numcol:pl:gen:m1:rec', ''),
    ('dwoje', 'numcol:pl:acc:m1:rec', ''),
    ('czworo', 'numcol:pl:nom:m1:rec', ''),
    ('hej', 'interj', ''),
    ('jeszcze', 'qub', 'part'),
    ('czterem', 'num:pl:dat:m1:congr', ''),
    ('czym', 'conj', 'comp'),
    ('niedaleko', 'prep:gen', ''),
    ('doprawdy', 'qub', 'adv'),
    ('jak', 'qub', 'adv'),
    ('pół', '', 'numcomp'),
    ('pół', '', 'num:comp'),
    ('pół', 'num:pl:acc:n:rec', ''),
    ('słowa', 'subst:pl:acc:n', 'subst:sg:gen:n:ncol'),
    ('rozklepywało', '', 'praet:sg:n1:ter:imperf'),
    ('bardzo', 'adv:pos', 'adv'),
    ('bardziej', 'adv:com', ''),
    ('znacząco', 'adv:pos', 'pacta'),
    ('my', '', 'ppron12:pl:nom:_:pri'),
    ('sobie', 'siebie:dat', ''),
    ('zł', 'brev:npun', 'brev'),
]

@pytest.mark.parametrize('form, exist, not_exist', test_data)
def test_maca(maca, form, exist, not_exist):
    paragraph=maca.analyze(form)
    sentence=paragraph.sentences[0]
    token=sentence.tokens[0]
    tags = [form.tags for form in token.interpretations]
    print(tags)
    if exist:
        assert exist in tags
    if not_exist:
        assert not_exist not in tags

@pytest.mark.parametrize('form, exist, not_exist', test_data)
def test_morfeusz(maca, form, exist, not_exist):
    morfeusz = get_morfeusz()
    tags=[tag for form, tag in analyze_token(morfeusz, form)]
    print(tags)
    if exist:
        assert exist in tags
    if not_exist:
        assert not_exist not in tags