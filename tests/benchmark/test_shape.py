import pytest

from krnnt.new import shape
import krnnt_utils

@pytest.fixture
def word():
    return "ljhbasjk8f5IYTVIGHVaisftityvfiouyfO*86f97f697"

def test_shape_regex(word, benchmark):
    benchmark(shape,word)

def test_shape_cython(word, benchmark):
    benchmark(krnnt_utils.shape,word)