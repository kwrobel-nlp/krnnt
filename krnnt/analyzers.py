import re
import sys
from subprocess import PIPE, Popen
from typing import Iterable

from .classes import Form, Token, Sentence, Paragraph
from krnnt.new import uniq

try:
    from maca_analyse import maca_analyse
except ImportError:
    pass


# TODO morfeusz analyzer for pretokenized?

class MacaAnalyzer:
    def __init__(self, maca_config: str, toki_config_path: str = ''):
        self.maca_config = maca_config
        self.toki_config_path = toki_config_path
        self.configure_maca()

    def configure_maca(self):
        if 'maca_analyse' in sys.modules:
            self._maca = self._maca_wrapper
        else:
            self._maca = self._maca_process

    def analyze(self, text: str) -> Paragraph:
        results = self._maca([text])

        paragraph_reanalyzed = Paragraph()
        for i, res in enumerate(results):
            result = self._parse(res)
            sentence_reanalyzed = Sentence()
            paragraph_reanalyzed.add_sentence(sentence_reanalyzed)
            for form, space_before, interpretations, start, end in result:
                token_reanalyzed = Token()
                sentence_reanalyzed.add_token(token_reanalyzed)
                token_reanalyzed.form = form
                token_reanalyzed.space_before = space_before  # != 'none'
                interpretations = [(re.sub(r':[abcdijnopqsv]\d?$', '', l), t) for l, t in
                                   interpretations]  # remove senses
                token_reanalyzed.interpretations = [Form(l.replace('_', ' '), t) for l, t in uniq(interpretations)]
                token_reanalyzed.start = start
                token_reanalyzed.end = end
        return paragraph_reanalyzed

    def _maca_process(self, batch: Iterable[str]):
        cmd = ['maca-analyse', '-c', self.maca_config, '-l']
        if self.toki_config_path:
            cmd.extend(['--toki-config-path', self.toki_config_path])
        p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        self.text = '\n'.join(batch)
        self.last_offset = 0

        stdout = p.communicate(input=self.text.encode('utf-8'))[0]
        try:
            p.stdin.close()
        except BrokenPipeError:
            pass
        p.wait()
        if p.returncode != 0:
            raise Exception('Maca is not working properly')
        for i in stdout.decode('utf-8').split('\n\n'):
            if len(i) > 0:
                yield i

    def _maca_wrapper(self, batch: Iterable[str]):
        self.text = '\n'.join(batch)
        self.last_offset = 0

        output_text = maca_analyse(self.maca_config, self.toki_config_path, self.text)

        for i in output_text.split('\n\n'):
            if len(i) > 0:
                yield i

    def _parse(self, output):
        data = []
        lemma_lines = []
        token_line = None
        for line in output.split("\n"):
            if line.startswith("\t"):
                lemma_lines.append(line)
            else:
                if token_line is not None:
                    data.append((token_line, lemma_lines))
                    lemma_lines = []
                token_line = line
        data.append((token_line, lemma_lines))

        tokens = []

        for index, (token_line, lemma_lines) in enumerate(data):
            token = self._construct(token_line, lemma_lines)  # 80%
            if token is None: continue
            form, space_before, interpretations = token
            start = self.text.index(form, self.last_offset)
            end = start + len(form)
            self.last_offset = end
            tokens.append((form, space_before, interpretations, start, end))

        return tokens

    def _construct(self, token_line, lemma_lines):
        try:
            if token_line == '': return None
            form, space_before = token_line.split("\t")
        except ValueError:
            raise Exception('Probably Maca is not working.')

        interpretations = []

        for lemma_line in lemma_lines:
            row = lemma_line.strip().split("\t")
            try:
                lemma, tags, _ = row  # 30%
                # disamb = True
            except ValueError:
                lemma, tags = row  # 16%
                # disamb = False
            interpretation = (lemma, tags)
            # lemma.disamb=disamb
            interpretations.append(interpretation)

        return form, space_before, interpretations
