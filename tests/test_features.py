import pytest

from krnnt.features import FeaturePreprocessor, TagsPreprocessorCython, TagsPreprocessor, create_token_features


@pytest.fixture
def token():
    return 'asd'


def test_nic(token):
    assert ["NIC"] == FeaturePreprocessor.nic(token)


@pytest.mark.xfail
@pytest.mark.parametrize('token, expected', [('kot', 'islower'),
                                             ('Kot', 'istitle'),
                                             ('KOT', 'isupper'),
                                             ('2019', 'isdigit'),
                                             ('Kot:)', 'ismixed'),
                                             ('kot:)', 'ismixed'),
                                             ('KOT:)', 'ismixed'),
                                             ('2019:)', 'ismixed')])
def test_cases(token, expected):
    features = FeaturePreprocessor.cases(token)
    assert features[0] == expected
    assert len(features) == 1


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


@pytest.mark.xfail
def test_qubliki():
    assert [] == FeaturePreprocessor.qubliki('kot')
    assert ['ale'] == FeaturePreprocessor.qubliki('ale')
    assert ['ale'] == FeaturePreprocessor.qubliki('Ale')


@pytest.mark.parametrize('token, expected', [('wrobel', 'l'),
                                             ('Wrobel', 'ul'),
                                             ('WROBEL', 'u'),
                                             ('2019', 'd'),
                                             ('Wrobel2019', 'uld'),
                                             ('Wrobel2019:)', 'uldx')])
def test_shape(token, expected):
    features = FeaturePreprocessor.shape(token)
    assert features[0] == expected
    assert len(features) == 1


@pytest.mark.parametrize('tags, expected', [
    (['fin:sg:ter:imperf', 'subst:sg:nom:f'], ['1fin:ter', '2fin:sg:imperf', '1subst:nom',
                                               '2subst:sg:f']),
    (['interp'], ['1interp', '2interp']),
    ([''], ['1', '2']),
    ([], [])])
def test_tags4(tags, expected):
    assert TagsPreprocessor.create_tags4_without_guesser(tags) == expected
    assert TagsPreprocessorCython.create_tags4_without_guesser(tags) == expected


@pytest.mark.parametrize('tags, expected', [
    (['fin:sg:ter:imperf', 'subst:sg:nom:f'], ['sg', 'sg:nom:f', 'nom']),
    (['interp'], []),
    ([''], []),
    ([], [])])
def test_tags5(tags, expected):
    assert TagsPreprocessor.create_tags5_without_guesser(tags) == expected
    assert TagsPreprocessorCython.create_tags5_without_guesser(tags) == expected


def test_create_token_features(benchmark):
    token = 'obejmie'
    tags = ['subst:sg:loc:m3', 'subst:sg:voc:m3', 'subst:sg:dat:f', 'subst:sg:loc:f',
            'fin:sg:ter:perf']
    space_before = ['space_before']
    features=['islower', 'l', 'P0o', 'P1b', 'P2e', 'S1e', 'S2i', 'S3m', '1subst:loc', '2subst:sg:m3',
                          '1subst:voc', '1subst:dat', '2subst:sg:f', '1fin:ter', '2fin:sg:perf', 'sg:loc:m3', 'loc',
                          'sg:voc:m3', 'voc', 'sg:dat:f', 'dat', 'sg:loc:f', 'sg', 'space_before']

    result_features = create_token_features(token, tags, space_before)
    assert result_features == features
