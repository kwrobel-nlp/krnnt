import pytest

from krnnt.features import FeaturePreprocessor, TagsPreprocessorCython, TagsPreprocessor


@pytest.fixture
def token():
    return 'asd'


def test_nic(token):
    print(FeaturePreprocessor.nic(token))
    assert ["NIC"] == FeaturePreprocessor.nic(token)


def test_cases():
    assert ["islower"] == FeaturePreprocessor.cases('kot')
    assert ["istitle"] == FeaturePreprocessor.cases('Kot')
    assert ["isupper"] == FeaturePreprocessor.cases('KOT')
    assert ["isdigit"] == FeaturePreprocessor.cases('2019')
    assert ["ismixed"] == FeaturePreprocessor.cases('Kot:)')
    assert ["ismixed"] == FeaturePreprocessor.cases('kot:)')
    assert ["ismixed"] == FeaturePreprocessor.cases('KOT:)')
    assert ["ismixed"] == FeaturePreprocessor.cases('2019:)')


def test_interps():
    assert ["."] == FeaturePreprocessor.interps('.', {'tags': ['interp']})
    assert [] == FeaturePreprocessor.interps('.', {'tags': ['subst']})
    assert [] == FeaturePreprocessor.interps(':)', {'tags': ['interp']})


def test_prefix1():
    assert ["P0k"] == FeaturePreprocessor.prefix1('kot')
    assert ["P0??"] == FeaturePreprocessor.prefix1('©kot')
    assert ["P0k"] == FeaturePreprocessor.prefix1('KOT')


def test_prefix2():
    assert ["P1o"] == FeaturePreprocessor.prefix2('kot')
    assert ["P1xx"] == FeaturePreprocessor.prefix2('k')


def test_prefix3():
    assert ["P2t"] == FeaturePreprocessor.prefix3('kot')


def test_suffix1():
    assert ["S1t"] == FeaturePreprocessor.suffix1('kot')
    assert ["S1??"] == FeaturePreprocessor.suffix1('kot©')


def test_suffix2():
    assert ["S2o"] == FeaturePreprocessor.suffix2('kot')
    assert ["S2xx"] == FeaturePreprocessor.suffix2('k')


def test_suffix3():
    assert ["S3k"] == FeaturePreprocessor.suffix3('kot')


def test_qubliki():
    assert [] == FeaturePreprocessor.qubliki('kot')
    assert ['ale'] == FeaturePreprocessor.qubliki('ale')
    assert ['ale'] == FeaturePreprocessor.qubliki('Ale')


def test_shape(token):
    assert ["l"] == FeaturePreprocessor.shape('wrobel')
    assert ["ul"] == FeaturePreprocessor.shape('Wrobel')
    assert ["u"] == FeaturePreprocessor.shape('WROBEL')
    assert ["d"] == FeaturePreprocessor.shape('2019')
    assert ["uld"] == FeaturePreprocessor.shape('Wrobel2019')
    assert ["uldx"] == FeaturePreprocessor.shape('Wrobel2019:)')


def test_tags4():
    input = ['fin:sg:ter:imperf', 'subst:sg:nom:f']
    out = ['1fin:ter', '2fin:sg:imperf', '1subst:nom',
           '2subst:sg:f']
    assert out == TagsPreprocessor.create_tags4_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags4_without_guesser(input)

    input = ['interp']
    out = ['1interp', '2interp']
    assert out == TagsPreprocessor.create_tags4_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags4_without_guesser(input)

    input = ['']
    out = ['1', '2']
    assert out == TagsPreprocessor.create_tags4_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags4_without_guesser(input)

    input = []
    out = []
    assert out == TagsPreprocessor.create_tags4_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags4_without_guesser(input)

def test_tags5():
    input = ['fin:sg:ter:imperf', 'subst:sg:nom:f']
    out = ['sg', 'sg:nom:f', 'nom']
    assert out == TagsPreprocessor.create_tags5_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags5_without_guesser(input)

    input = ['interp']
    out = []
    assert out == TagsPreprocessor.create_tags5_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags5_without_guesser(input)

    input = ['']
    out = []
    assert out == TagsPreprocessor.create_tags5_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags5_without_guesser(input)

    input = []
    out = []
    assert out == TagsPreprocessor.create_tags5_without_guesser(input)
    assert out == TagsPreprocessorCython.create_tags5_without_guesser(input)