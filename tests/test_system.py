import pytest


def test_process_xces(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'python3 process_xces.py tests/data/small/nkjp1m-1.2-xces.xml /tmp/nkjp.spickle',
        'diff /tmp/nkjp.spickle tests/data/reference/nkjp1m-1.2.spickle']

    for command in commands:
        bash.run_script_inline([command])


def test_reanalyze(bash, rootdir):
    # version of morfeusz dictionary may influence results
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'python3 reanalyze.py --maca_config $MACA_CONFIG /tmp/nkjp.spickle /tmp/nkjp-reanalyzed.spickle',
        'diff /tmp/nkjp-reanalyzed.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.spickle'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp'}) as s:
        for command in commands:
            print(command)
            s.run_script_inline([command])


def test_shuffle(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'python3 shuffle.py tests/data/reference/nkjp1m-1.2-reanalyzed.spickle /tmp/nkjp-reanalyzed.shuf.spickle',
        'diff /tmp/nkjp-reanalyzed.shuf.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp'}) as s:
        for command in commands:
            s.run_script_inline([command])

@pytest.mark.slow
def test_train(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        # 'rm /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2 /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues',
        'python3 krnnt_train.py --maca_config $MACA_CONFIG /tmp/nkjp-reanalyzed.shuf.spickle -e 2 --reproducible --hash test',

        'h5diff weight_test.hdf5 tests/data/reference/weight_test.hdf5',
        'h5diff weight_test.hdf5.final tests/data/reference/weight_test.hdf5.final',
        'diff lemmatisation_test.pkl tests/data/reference/lemmatisation_test.pkl',
        'diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2 tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2',
        'diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData',
        'diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues',
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0'}) as s:
        for command in commands:
            s.run_script_inline([command])


def test_run_xces(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces > /tmp/out.xces',
        'diff /tmp/out.xces tests/data/reference/out.xces'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':''}) as s:
        for command in commands:
            s.run_script_inline([command])


def test_run_plain(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o plain > /tmp/out.plain',
        'diff /tmp/out.plain tests/data/reference/out.plain'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':''}) as s:
        for command in commands:
            s.run_script_inline([command])


def test_run_conll(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o conll > /tmp/out.conll',
        'diff /tmp/out.conll tests/data/reference/out.conll'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':''}) as s:
        for command in commands:
            s.run_script_inline([command])


def test_run_conllu(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o conllu > /tmp/out.conllu',
        'diff /tmp/out.conllu tests/data/reference/out.conllu'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':''}) as s:
        for command in commands:
            s.run_script_inline([command])


def test_run_jsonl(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',

        'echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o jsonl > /tmp/out.jsonl',
        'diff /tmp/out.jsonl tests/data/reference/out.jsonl'
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':''}) as s:
        for command in commands:
            s.run_script_inline([command])

def test_run_evaluation(bash, rootdir):
    commands = [
        'cd %s' % rootdir,
        'cd ..',
        'cat tests/data/small/gold-task-c.txt | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces --reproducible > /tmp/out.xces',
        'python2 tagger-eval.py  tests/data/small/gold-task-c.xml /tmp/out.xces > /tmp/out_evaluation.txt',
        'diff /tmp/out_evaluation.txt tests/data/reference/gold-task-c_evaluation.txt '
    ]

    with bash(envvars={'MACA_CONFIG': 'morfeusz2-nkjp', 'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0'}) as s:
        for command in commands:
            s.run_script_inline([command])