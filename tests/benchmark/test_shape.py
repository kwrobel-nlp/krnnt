import pytest

from krnnt.new import shape, shape2, shape3


@pytest.fixture
def word():
    return "ljhbasjk8f5IYTVIGHVaisftityvfiouyfO*86f97f697"

def test_shape2(word, benchmark):
    benchmark(shape2,word)

def test_shape3(word, benchmark):
    benchmark(shape3,word)

def test_shape_precompiled(word, benchmark):
    benchmark(shape,word)
