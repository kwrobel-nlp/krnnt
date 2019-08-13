from krnnt.writers import results_to_conll_str, results_to_conllu_str, results_to_txt_str, results_to_plain_str, \
    results_to_xces_str

results = [[{'token': 'Lubię', 'sep': 'newline', 'prob': 0.37375012, 'tag': 'adj:pl:nom:m1:pos', 'lemmas': ['Lubię'],
             'start': 0, 'end': 5},
            {'token': 'placki', 'sep': 'space', 'prob': 0.38550463, 'tag': 'subst:pl:nom:m1', 'lemmas': ['placki'],
             'start': 6, 'end': 12},
            {'token': '.', 'sep': 'none', 'prob': 0.99999726, 'tag': 'interp', 'lemmas': ['.'], 'start': 12,
             'end': 13}], [
               {'token': 'Ala', 'sep': 'space', 'prob': 0.9995969, 'tag': 'subst:sg:nom:f', 'lemmas': ['Ala'],
                'start': 14, 'end': 17},
               {'token': 'ma', 'sep': 'space', 'prob': 0.6605565, 'tag': 'subst:sg:nom:f', 'lemmas': ['ma'],
                'start': 18, 'end': 20},
               {'token': 'kota', 'sep': 'space', 'prob': 0.93132496, 'tag': 'subst:sg:nom:f', 'lemmas': ['kota'],
                'start': 21, 'end': 25},
               {'token': '.', 'sep': 'none', 'prob': 0.9999993, 'tag': 'interp', 'lemmas': ['.'], 'start': 25,
                'end': 26}], [
               {'token': 'Raz', 'sep': 'space', 'prob': 0.23650545, 'tag': 'subst:sg:nom:f', 'lemmas': ['Raz'],
                'start': 27, 'end': 30},
               {'token': 'dwa', 'sep': 'space', 'prob': 0.581044, 'tag': 'adj:pl:acc:f:pos', 'lemmas': ['dwa'],
                'start': 31, 'end': 34},
               {'token': 'trzy', 'sep': 'space', 'prob': 0.71970826, 'tag': 'subst:pl:acc:f', 'lemmas': ['trzy'],
                'start': 35, 'end': 39},
               {'token': '.', 'sep': 'none', 'prob': 0.99999905, 'tag': 'interp', 'lemmas': ['.'], 'start': 39,
                'end': 40}]]


def test_conll():
    reference=\
"""Lubię	Lubię	1	adj:pl:nom:m1:pos	0	5
placki	placki	1	subst:pl:nom:m1	6	12
.	.	0	interp	12	13

Ala	Ala	1	subst:sg:nom:f	14	17
ma	ma	1	subst:sg:nom:f	18	20
kota	kota	1	subst:sg:nom:f	21	25
.	.	0	interp	25	26

Raz	Raz	1	subst:sg:nom:f	27	30
dwa	dwa	1	adj:pl:acc:f:pos	31	34
trzy	trzy	1	subst:pl:acc:f	35	39
.	.	0	interp	39	40
"""
    output = results_to_conll_str(results)
    assert output == reference

def test_conllu():
    reference=\
"""1	Lubię	Lubię	_	adj:pl:nom:m1:pos	_	_	_	_	_
2	placki	placki	_	subst:pl:nom:m1	_	_	_	_	_
3	.	.	_	interp	_	_	_	_	_

1	Ala	Ala	_	subst:sg:nom:f	_	_	_	_	_
2	ma	ma	_	subst:sg:nom:f	_	_	_	_	_
3	kota	kota	_	subst:sg:nom:f	_	_	_	_	_
4	.	.	_	interp	_	_	_	_	_

1	Raz	Raz	_	subst:sg:nom:f	_	_	_	_	_
2	dwa	dwa	_	adj:pl:acc:f:pos	_	_	_	_	_
3	trzy	trzy	_	subst:pl:acc:f	_	_	_	_	_
4	.	.	_	interp	_	_	_	_	_
"""
    output = results_to_conllu_str(results)
    assert output == reference

def test_txt():
    reference=\
"""Lubię placki.
Ala ma kota.
Raz dwa trzy.
"""
    output = results_to_txt_str(results)

    assert output == reference

def test_plain():
    reference=\
"""Lubię	newline
	Lubię	adj:pl:nom:m1:pos	disamb
placki	space
	placki	subst:pl:nom:m1	disamb
.	none
	.	interp	disamb

Ala	space
	Ala	subst:sg:nom:f	disamb
ma	space
	ma	subst:sg:nom:f	disamb
kota	space
	kota	subst:sg:nom:f	disamb
.	none
	.	interp	disamb

Raz	space
	Raz	subst:sg:nom:f	disamb
dwa	space
	dwa	adj:pl:acc:f:pos	disamb
trzy	space
	trzy	subst:pl:acc:f	disamb
.	none
	.	interp	disamb
"""
    output = results_to_plain_str(results)
    assert output == reference

def test_xces():
    reference=\
"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">
<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">
<chunkList>
 <chunk type="p">
  <chunk type="s">
   <tok>
    <orth>Lubię</orth>
    <lex disamb="1"><base>Lubię</base><ctag>adj:pl:nom:m1:pos</ctag></lex>
   </tok>
   <tok>
    <orth>placki</orth>
    <lex disamb="1"><base>placki</base><ctag>subst:pl:nom:m1</ctag></lex>
   </tok>
   <ns/>
   <tok>
    <orth>.</orth>
    <lex disamb="1"><base>.</base><ctag>interp</ctag></lex>
   </tok>
  </chunk>
 </chunk>
 <chunk type="p">
  <chunk type="s">
   <tok>
    <orth>Ala</orth>
    <lex disamb="1"><base>Ala</base><ctag>subst:sg:nom:f</ctag></lex>
   </tok>
   <tok>
    <orth>ma</orth>
    <lex disamb="1"><base>ma</base><ctag>subst:sg:nom:f</ctag></lex>
   </tok>
   <tok>
    <orth>kota</orth>
    <lex disamb="1"><base>kota</base><ctag>subst:sg:nom:f</ctag></lex>
   </tok>
   <ns/>
   <tok>
    <orth>.</orth>
    <lex disamb="1"><base>.</base><ctag>interp</ctag></lex>
   </tok>
  </chunk>
 </chunk>
 <chunk type="p">
  <chunk type="s">
   <tok>
    <orth>Raz</orth>
    <lex disamb="1"><base>Raz</base><ctag>subst:sg:nom:f</ctag></lex>
   </tok>
   <tok>
    <orth>dwa</orth>
    <lex disamb="1"><base>dwa</base><ctag>adj:pl:acc:f:pos</ctag></lex>
   </tok>
   <tok>
    <orth>trzy</orth>
    <lex disamb="1"><base>trzy</base><ctag>subst:pl:acc:f</ctag></lex>
   </tok>
   <ns/>
   <tok>
    <orth>.</orth>
    <lex disamb="1"><base>.</base><ctag>interp</ctag></lex>
   </tok>
  </chunk>
 </chunk>
</chunkList>
</cesAna>"""

    output = results_to_xces_str(results)
    assert output == reference