from xml.etree import ElementTree as ET

import jsonlines

from krnnt.classes import Paragraph, Sentence, Token, Form


def read_xces(file_path):
    paragraphs_defined = True
    ns=False #no separator
    first_chunk=True
#    for event, elem in ET.iterparse(file_path, events=("start",)):
#        if elem.tag == 'chunk':
#            if elem.get('type') == 's':
#                paragraphs_defined = False
#            break

    # print('Paragrpahs defined:', paragraphs_defined)

    for event, elem in ET.iterparse(file_path, events=("start","end",)):
        if first_chunk and event=="start" and elem.tag in ('chunk','sentence'):
            if elem.get('type') == 's' or elem.tag =='sentence':
                paragraphs_defined = False
            first_chunk=False
        elif event=="end" and elem.tag in ('chunk','sentence'):
            xml_sentences=[]
            paragraph=Paragraph()
            if paragraphs_defined and elem.tag == 'chunk' and elem.get('type')!='s':
                xml_sentences = elem.getchildren()
            elif (not paragraphs_defined) and ((elem.tag == 'chunk' and elem.get('type')=='s') or elem.tag == 'sentence'):
                xml_sentences = [elem]
            else:
                continue

            for sentence_index, xml_sentence in enumerate(xml_sentences):
                sentence=Sentence()
                paragraph.add_sentence(sentence)
                for token_index, xml_token in enumerate(xml_sentence.getchildren()):
                    if xml_token.tag=='ns':
                        if token_index>0 or sentence_index>0: #omit first ns in paragraph
                            ns=True
                    elif xml_token.tag=='tok':
                        token=Token()
                        token.space_before=not ns

                        for xml_node in xml_token.getchildren():
                            if xml_node.tag=='orth':
                                orth=xml_node.text
                                token.form=orth
                            elif xml_node.tag=='lex':
                                if xml_node.get('disamb')=='1':
                                    disamb=True

                                else:
                                    disamb=False

                                base=xml_node.find('base').text
                                ctag=xml_node.find('ctag').text

                                form = Form(base, ctag)
                                if disamb:
                                    token.gold_form=form
                                else:
                                    token.interpretations.append(form)
                            elif xml_node.tag=='ann':
                                continue
                            else:
                                print('Error 1', xml_token)
                        if token.form:
                            sentence.add_token(token)
                        ns=False
                    else:
                        print('Error 2', xml_token)
            yield paragraph
            elem.clear()


def read_jsonl(file_path):
    with jsonlines.Reader(file_path) as reader:
        for obj in reader:
            a = list_to_paragraph(obj)
            yield a


def list_to_paragraph(l):
    paragraph = Paragraph()
    for s in l:
        sentence = Sentence()
        paragraph.add_sentence(sentence)
        for t in s:
            token = Token()
            form=t[0]
            token.form = form

            print(t)
            try:
                space=t[1]
                token.space_before = (space == 1)
            except IndexError:
                token.space_before = True # ?

            interpretations = t[2:]
            token.interpretations.extend([Form(base, ctag) for (base, ctag) in interpretations])

            sentence.add_token(token)
    return paragraph