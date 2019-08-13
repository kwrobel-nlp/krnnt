import pickle
from typing import BinaryIO, Iterable


class SerialPickler:
    def __init__(self, file: BinaryIO, mode=3):  # don't work with protocol 4
        self.file = file
        self.p = pickle.Pickler(file, mode)

    def add(self, obj):
        self.p.dump(obj)
        self.p.memo.clear()

    def extend(self, objs: Iterable):
        for obj in objs:
            self.p.dump(obj)
            self.p.memo.clear()

    def close(self):
        self.file.close()


class SerialUnpickler:
    def __init__(self, file: BinaryIO, stop: int=-1, start: int =0, ids: Iterable = None):
        """

        :param file:
        :param start: unpickle objects starting from index start
        :param stop: unpickle objects ending with index stop
        :param ids: unpickle objects with indexes in ids
        """
        if ids is None:
            ids = []
        self.file = file
        self.p = pickle.Unpickler(file)
        self.c = 0
        self.stop = stop
        self.start = start
        self.ids = set(ids)

    def __iter__(self):
        if self.ids:
            return self.__iter2()
        else:
            return self.__iter1()

    def __iter1(self):
        while True:
            try:
                if self.c == self.stop:
                    break
                self.c += 1
                x = self.p.load()
                if self.c - 1 < self.start:
                    continue

                # print self.c
                yield x
            except EOFError:
                break

    def __iter2(self):
        while True:
            try:
                x = self.p.load()
                if self.c in self.ids:
                    yield x
                self.c += 1
            except EOFError:
                break


def count_samples(path: str) -> int:
    """
    Return number of items in serial pickle file.
    """
    with open(path, 'rb') as file:
        su = SerialUnpickler(file)

        count = 0
        for paragraph in su:
            count += 1

        return count