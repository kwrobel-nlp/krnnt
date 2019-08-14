import pytest

from krnnt.features import TagsPreprocessor, TagsPreprocessorCython


@pytest.fixture
def tags():
    return ['fin:sg:ter:imperf', 'subst:sg:nom:f']


@pytest.mark.slow
def test_tags4(tags, benchmark):
    benchmark(TagsPreprocessor.create_tags4_without_guesser, tags)


@pytest.mark.slow
def test_tags4_cython(tags, benchmark):
    benchmark(TagsPreprocessorCython.create_tags4_without_guesser, tags)


@pytest.mark.slow
def test_tags5(tags, benchmark):
    benchmark(TagsPreprocessor.create_tags5_without_guesser, tags)


@pytest.mark.slow
def test_tags5_cython(tags, benchmark):
    benchmark(TagsPreprocessorCython.create_tags5_without_guesser, tags)
