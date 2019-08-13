import sys

from krnnt.structure import Token, Sentence, Paragraph


def text(buffer):
    return ''.join([' '+token.form if token.space_before else token.form for token in buffer])

def align(pred, ref, ref_text_old=''):
    pred_buffer = [pred.pop(0)]
    ref_buffer = [ref.pop(0)]
    if ref_text_old:
        t = Token()
        t.form = ref_text_old
        t.space_before=False
        ref_buffer.insert(0, t)

    while pred_buffer or ref_buffer:
        pred_text = text(pred_buffer)
        ref_text = text(ref_buffer)
        # print(pred_text, ref_text)
        if len(pred_text) == len(ref_text):  # aligned
            if pred_text != ref_text:
                print('alignment ERROR', pred_text, ref_text, ref, pred, file=sys.stderr)

            yield (pred_buffer, ref_buffer, ref_text[len(pred_text):])

            pred_buffer=[]
            ref_buffer = []

            #print(pred)
            if not pred or not ref:
                #print('break', pred)
                break

            pred_buffer = [pred.pop(0)]
            ref_buffer = [ref.pop(0)]
        elif len(pred_text) < len(ref_text):
            if pred:
                pred_buffer.append(pred.pop(0))
            else:

                print('break2', pred_text, ref_text, ref_text[len(pred_text):])
                #skroc ref_buffer
                asd=[]
                for x in ref_buffer:
                    asd.append(x)
                    if len(pred_text) >= len(text(asd)):

                        break
                ref_buffer=asd
                if len(pred_text) < len(text(asd)):
                    print('RRRR', asd[-1].form)
                    asd[-1].form = asd[-1].form[:len(pred_text)-1]
                    print('RRRR', asd[-1].form)
                print(text(ref_buffer), 'XXX', text(ref))


                break
        else:
            if ref:
                ref_buffer.append(ref.pop(0))
            else:
                print('break3')
                break

    rest = ref_buffer # + ref
    if rest:
        yield (pred_buffer+pred, rest, ref_text[len(pred_text):])
    # print('rest', pred, ref)

def align_paragraphs(paragraph_reanalyzed: Paragraph, paragraph_gold: Paragraph) -> Paragraph:
    tokens_gold = []
    for sentence_gold in paragraph_gold:
        for token_gold in sentence_gold:
            tokens_gold.append(token_gold)
            token_gold.form = token_gold.form.replace('\xa0', ' ')  # "a j e n t a"


    ref_text_old = ''
    paragraph_reanalyzed.concraft = []
    for sentence_reanalyzed in paragraph_reanalyzed:
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXNEW')
        sentence_reanalyzed_gold = Sentence()
        paragraph_reanalyzed.concraft.append(sentence_reanalyzed_gold)
        for p, r, ref_text_old in align([token for token in sentence_reanalyzed.tokens], tokens_gold, ref_text_old):

            if p:
                for r1 in r:
                    sentence_reanalyzed_gold.add_token(r1)
                if text(p) != text(r):
                    print('ERR', [t.form for t in p], [t.form for t in r])
                # if len(p)!=len(r):
                # print(text(p),'_____', text(r))
                # print(len(tokens_gold))
                if len(p) == len(r):
                    for p1, r1 in zip(p, r):
                        p1.gold_form = r1.gold_form
    return paragraph_reanalyzed