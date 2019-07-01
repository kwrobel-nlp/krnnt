import collections

import sys

from krnnt.writers import results_to_xces, results_to_xces_str
from krnnt.readers import read_xces

# path='/home/djstrong/projects/repos/krnnt/models/voting/'
# path='/home/djstrong/projects/repos/krnnt/'
# files=[path+'text-raw.'+str(i)+'.xml' for i in range(4)]
# files=[path+str(i)+'b.xml' for i in range(10)]

path=sys.argv[1]
files=[path+str(i)+'.xml' for i in range(10)]

def checkEqual2(iterator):
   return len(set(iterator)) == 1

xcess = [read_xces(file) for file in files]

result = []

count_all=0
count_mismatch=0

while True:
    try:
        paragraphs = [next(xces) for xces in xcess]

        for sentences in zip(*paragraphs):
            sentence = []
            result.append(sentence)
            for tokens in zip(*sentences):
                count_all+=1
                # print(tokens)
                forms = [token.gold_form for token in tokens]
                tags = [form.tags for form in forms]

                token_result = {'sep': 'space' if tokens[0].space_before else 'none','token':tokens[0].form}
                sentence.append(token_result)
                if not checkEqual2(tags):
                    # print(tags)
                    tags_count=collections.defaultdict(list)
                    for form in forms:
                        tags_count[form.tags].append(form)
                    # print(tags_count)

                    sorted_forms = sorted(tags_count.items(), key=lambda x: len(x[1]), reverse=True)
                    # print(tokens[0].form, '\t'*(3-int(len(tokens[0].form)/8)), [(form[0], len(form[1])) for form in sorted_forms])
                    winner = sorted_forms[0][1][0]



                    token_result['tag']=winner.tags
                    token_result['lemmas']=[winner.lemma]
                    count_mismatch+=1
                else:
                    # print(tokens[0].form, '\t'*(3-int(len(tokens[0].form)/8)), forms[0].tags)
                    token_result['tag']=forms[0].tags
                    token_result['lemmas']=[forms[0].lemma]

            # print()
        # print()



    except StopIteration:
        break


print(results_to_xces_str(result))

print(count_all, count_mismatch, file=sys.stderr)