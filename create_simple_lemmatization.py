import sys

sgjp=sys.argv[1]

print(sgjp)

def base_tag(tag):
    transformations = {
          'ger':  [(['pl'],'sg'), 
                   (['gen','dat','acc','inst','loc','voc'],'nom')], 
          'pact': [(['pl'],'sg'), 
                   (['gen','dat','acc','inst','loc','voc'],'nom'), 
                   (['m2','m3','f','n'], 'm1')],
          'ppas': [(['pl'],'sg'), 
                   (['gen','dat','acc','inst','loc','voc'],'nom'), 
                   (['m2','m3','f','n'], 'm1')],
    }
    
    tag=list(tag)
    
    if tag[0] not in transformations: return None
    
    transforms = transformations[tag[0]]
    for sources, target in transforms:
        for source in sources:
            try:
                index = tag.index(source)
                tag[index]=target
                break
            except ValueError:
                pass
    return tag
  
  
  

import itertools
import tqdm

count=0


lt={}
for line in tqdm.tqdm(open(sgjp), total=7221123):
    row = line.split('\t')[:-1]
#     print(row)
    try:
        form, lemma, tag, other = row
    except ValueError:
        continue
    
    tags=[t.split('.') for t in tag.split(':')]
    for tag in itertools.product(*tags):
        if tag[0] in ['ger','ppas','pact']:
            btag=tuple(base_tag(tag))
#             print(tag, btag, form, lemma)
            if btag == tag:
                count+=1
                lemma=lemma.rsplit(':')[0]
                lt[(lemma,tag)]=form

print(count)

import pickle
pickle.dump(lt, open('data/ger_ppas_pact.pickle','wb'))