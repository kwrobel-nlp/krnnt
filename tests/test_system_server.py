def test_download_model(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        './download_model.sh'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])

#TODO: run server: python3 krnnt_serve.py model_data/ --maca_config morfeusz2-nkjp

def test_post_raw(bash,rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'curl -X POST "http://localhost:9200" --data-binary @tests/data/server/in_raw.txt  > /tmp/out.txt',
        'diff /tmp/out.txt tests/data/server/out_raw.txt'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])

def test_post_form(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'curl -X POST "http://localhost:9200" --data-binary "text=Lubię placki. Ala ma kota.\n\nRaz dwa trzy." > /tmp/out.txt'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])

    generated = open('/tmp/out.txt').read()
    reference = open('data/server/out_raw.txt').read()

    assert reference in generated

def test_post_tokenized_json(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'curl -X POST -H "Content-Type: application/json" "http://localhost:9200" -d @tests/data/server/in_tokenized.json > /tmp/out.txt',
        'diff /tmp/out.txt tests/data/server/out_raw.txt'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])

def test_post_tokenized_compact_json(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'curl -X POST -H "Content-Type: application/json" "http://localhost:9200" -d @tests/data/server/in_tokenized_compact.json > /tmp/out.txt',
        'diff /tmp/out.txt tests/data/server/out_raw.txt'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])

def test_post_raw_poleval(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'curl -X POST "http://localhost:9200" --data-binary @tests/data/full/test-raw.txt  > /tmp/out.txt'
    ]

    with bash() as s:
        for command in commands:
            s.run_script_inline([command])