import pytest

from krnnt.analyzers import MacaAnalyzer

paragraph_raw = 'Moje niefortunne pudła przygnębiły mnie do reszty. Zawsze miałem pretensje, że jestem dobrym myśliwym, a od dzieciństwa nie rozstawałem się ze strzelbą, a tu wśród obcych zblamowałem się jak nigdy w życiu. Jakże inaczej strzelałem cietrzewie i pardwy z moich "hollandów", które pozostawiłem na wieczną zgubę w Petersburgu. Poczciwy Staś Sierakowski pośpieszył mi z pomocą, by wyjaśnić moje niepowodzenia. - Pokaż mi strzelbę - poprosił, a gdy podałem mu mojego mauzera, spytał ze śmiechem: - Gdzieś to świństwo wykopał? - Ano w Gdańsku - odrzekłem zawstydzony. - Chyba byłeś ślepy, kupując taką szkaradę. Z czego strzelałeś przed wojną? - Miałem hollandy - odrzekłem. - Jedyna rada - rzekł w końcu Staś po oględzinach mojej broni. - Każ sobie skrócić szyję na dobrych kilka centymetrów, albo jeszcze lepiej rzuć to świństwo do pieca, a co się nie spali - na śmietnik.'
MACA_CONFIG1 = 'morfeusz-nkjp-official'
MACA_CONFIG2 = 'morfeusz2-nkjp'


@pytest.fixture
def get_maca_wrapper():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        list(maca_analyzer._maca_wrapper([paragraph_raw]))
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        list(maca_analyzer._maca_wrapper([paragraph_raw]))

    return maca_analyzer


@pytest.fixture
def get_maca_process():
    try:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG1)
        list(maca_analyzer._maca_process([paragraph_raw]))
    except:
        maca_analyzer = MacaAnalyzer(MACA_CONFIG2)
        list(maca_analyzer._maca_process([paragraph_raw]))

    return maca_analyzer


def analyze_process(maca_analyzer, data):
    results = maca_analyzer._maca_process(data)
    return list(results)


def analyze_wrapper(maca_analyzer, data):
    results = maca_analyzer._maca_wrapper(data)
    return list(results)


@pytest.mark.slow
def test_maca_process_speed(benchmark, get_maca_process):
    maca_analyzer = get_maca_process
    benchmark(analyze_process, maca_analyzer, [paragraph_raw])


@pytest.mark.slow
def test_maca_wrapper_speed(benchmark, get_maca_wrapper):
    maca_analyzer = get_maca_wrapper
    benchmark(analyze_wrapper, maca_analyzer, [paragraph_raw])
