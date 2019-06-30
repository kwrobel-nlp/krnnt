from .classes import Form, Token, uniq, Sentence, Paragraph
from .pipeline import Preprocess


class MacaAnalyzer:
    def __init__(self, maca_config: str):
        self.maca_config = maca_config

    def analyze(self, text: str) -> Paragraph:
        results = Preprocess.maca([text], maca_config=self.maca_config)
        results = list(results) #generator to list

        # tokens_reanalyzed = []

        paragraph_reanalyzed = Paragraph()
        for i, res in enumerate(results):
            result = Preprocess.parse(res)
            sentence_reanalyzed = Sentence()
            paragraph_reanalyzed.add_sentence(sentence_reanalyzed)
            for form, space_before, interpretations in result:
                token_reanalyzed = Token()
                sentence_reanalyzed.add_token(token_reanalyzed)
                # tokens_reanalyzed.append(token_reanalyzed)
                token_reanalyzed.form = form
                token_reanalyzed.space_before = space_before != 'none'
                token_reanalyzed.interpretations = [Form(l, t) for l, t in uniq(interpretations)]
        return paragraph_reanalyzed
