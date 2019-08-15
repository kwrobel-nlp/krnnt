from typing import Iterable, List

import regex


def unix_uniq(l: str) -> str:
    packed = []

    for el in l:
        if not packed or packed[-1] != el:
            packed.append(el)
    return ''.join(packed)


def uniq(seq: Iterable) -> List:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def flatten(l: Iterable) -> List:
    return [item for sublist in l for item in sublist]


def shape(word: str) -> str:  # TODO zredukowac czas
    word = regex.sub(r'(?V1)\p{Lowercase}', 'l', word, flags=regex.U)  # 80%
    word = regex.sub(r'(?V1)\p{Uppercase}', 'u', word, flags=regex.U)
    word = regex.sub(r'\p{gc=Decimal_Number}', 'd', word, flags=regex.U)
    word = regex.sub(r'[^A-Za-z0-9]', 'x', word, flags=regex.LOCALE)
    return unix_uniq(word)