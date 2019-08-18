import concurrent.futures
import os

import pytest
import requests


def test_api(rootdir):
    url = 'http://localhost:9200'

    for line in open(os.path.join(rootdir, 'data/full/test-raw.txt')):
        line=line.strip()
        if not line: continue

        tag('http://localhost:9200', line)

def tag(url, data):
    payload = data.encode('utf-8')
    r = requests.post(url, data=payload)
    return r

def chunk(l, batch_size):
    batch = []
    for element in l:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

@pytest.mark.parametrize('chunk_size', [100000, 10000, 1000, 100, 10, 4, 2,1])
def test_parallel_api(rootdir, chunk_size):
    print(rootdir, chunk_size)

    lines=[]
    for line in open(os.path.join(rootdir, 'data/full/test-raw.txt')):
        line = line.strip()
        if not line: continue
        lines.append(line)

    batches = list(chunk(lines, chunk_size))
    print(len(batches))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(tag, 'http://localhost:9200', "\n\n".join(batch)): "\n\n".join(batch) for batch in batches}
        for future in concurrent.futures.as_completed(future_to_url):
            r=future.result()
            # print(r.text)

def test_parallel_api_maca(rootdir):
    lines=[]
    for line in open(os.path.join(rootdir, 'data/full/test-raw.txt')):
        line = line.strip()
        if not line: continue
        lines.append(line)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(tag, 'http://localhost:9200/maca/', line): line for line in lines}
        for future in concurrent.futures.as_completed(future_to_url):
            r=future.result()
