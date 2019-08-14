import concurrent.futures
import os

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

def test_parallel_api(rootdir):
    lines=[]
    for line in open(os.path.join(rootdir, 'data/full/test-raw.txt')):
        line = line.strip()
        if not line: continue
        lines.append(line)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(tag, 'http://localhost:9200', line): line for line in lines}
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
        future_to_url = {executor.submit(tag, 'http://localhost:9200/maca', line): line for line in lines}
        for future in concurrent.futures.as_completed(future_to_url):
            r=future.result()
