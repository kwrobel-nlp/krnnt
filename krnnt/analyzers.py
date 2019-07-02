from subprocess import PIPE, Popen
from typing import Iterable

from .classes import Form, Token, uniq, Sentence, Paragraph



class MacaAnalyzer:
    def __init__(self, maca_config: str, toki_config_path: str = ''):
        self.maca_config = maca_config
        self.toki_config_path = toki_config_path

    def analyze(self, text: str) -> Paragraph:
        results = self._maca([text])
        results = list(results)  #TODO generator to list

        #TODO: start end of tokens

        paragraph_reanalyzed = Paragraph()
        for i, res in enumerate(results):
            result = self._parse(res)
            sentence_reanalyzed = Sentence()
            paragraph_reanalyzed.add_sentence(sentence_reanalyzed)
            for form, space_before, interpretations in result:
                token_reanalyzed = Token()
                sentence_reanalyzed.add_token(token_reanalyzed)
                token_reanalyzed.form = form
                token_reanalyzed.space_before = space_before != 'none'
                token_reanalyzed.interpretations = [Form(l, t) for l, t in uniq(interpretations)]
        return paragraph_reanalyzed


    def _maca(self, batch: Iterable[str]):
        cmd = ['maca-analyse', '-c', self.maca_config, '-l']
        if self.toki_config_path:
            cmd.extend(['--toki-config-path',self.toki_config_path])
        p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        stdout = p.communicate(input='\n'.join(batch).encode('utf-8'))[0]
        try:
          p.stdin.close()
        except BrokenPipeError:
          pass
        p.wait()
        if p.returncode!=0:
          raise Exception('Maca is not working properly')
        for i in stdout.decode('utf-8').split('\n\n'):
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
            token = self._construct(token_line, lemma_lines) #80%
            if token is None: continue
            tokens.append(token)

        return tokens


    def _construct(self, token_line, lemma_lines):
        try:
            if token_line == '': return None
            form, separator_before = token_line.split("\t")
        except ValueError:
            raise Exception('Probably Maca not working.')  # TODO what?

        form = form
        space_before = separator_before
        interpretations = []

        for lemma_line in lemma_lines:
            try:
                lemma, tags, _ = lemma_line.strip().split("\t")  # 30%
                # disamb = True
            except ValueError:
                lemma, tags = lemma_line.strip().split("\t")  # 16%
                # disamb = False
            lemma = (lemma, tags)
            # lemma.disamb=disamb
            interpretations.append(lemma)

        return form, space_before, interpretations