import logging


def startswith(token, prefixes):
    for prefix in prefixes:
        if token.lower().startswith(prefix.lower()):
            return True
    return False


def praet_or_winien(tag):
    return startswith(tag, ['praet', 'winien'])


def rule1(sentence):
    """
    Find immediate aglt after praet.
    """
    result = []

    for i, token in enumerate(sentence):

        separator = token['sep']
        tag = token['tag']

        if tag.startswith('aglt') and separator == 'none':
            if praet_or_winien(sentence[i - 1]['tag']):
                result.append((i, i - 1, None))
            elif praet_or_winien(sentence[i - 2]['tag']) and sentence[i - 1]['token'] == 'by':
                if sentence[i - 1]['sep'] == 'none':
                    result.append((i, i - 2, i - 1))
                else:
                    print('błąd?')
    return result

def rule1b(sentence):
    """
    Find immediate aglt after praet.
    """
    result = []

    for i, token in enumerate(sentence):


        tag = token['tag']

        if praet_or_winien(tag):
            try:
                next_token=sentence[i+1]
                if next_token['tag'].startswith('aglt') and next_token['sep'] == 'none':
                    result.append((i+1, i, None))
                elif next_token['tag']=='qub' and next_token['token'] == 'by' and next_token['sep'] == 'none':
                    try:
                        next_next_token=sentence[i+2]
                        if next_next_token['tag'].startswith('aglt') and next_next_token['sep'] == 'none':
                            result.append((i+2, i , i + 1))
                        else:
                            result.append((None, i, i + 1))
                    except IndexError:
                        result.append((None, i, i + 1))
            except IndexError:
                pass
    return result

def rule3(sentence):
    """
    Find aglt and then praet as successor.
    """
    result = []

    for i, token in enumerate(sentence):
        tag = token['tag']
        if tag.startswith('aglt'):
            for j in range(i + 1, len(sentence)):
                token2 = sentence[j]
                if praet_or_winien(token2['tag']):
                    by_index=None
                    try:
                        by_token = sentence[i-1]
                        if by_token['tag']=='qub' and by_token['token']=='by':
                            by_index=i-1
                    except IndexError:
                        pass
                    result.append((i, j, by_index))
                    # print(sentence[i - 2:j + 2])
                    break
        elif tag == 'qub' and token['token']=='by':
            try:
                if not sentence[i+1]['tag'].startswith('aglt'):
                    for j in range(i + 1, len(sentence)):
                        token2 = sentence[j]
                        if praet_or_winien(token2['tag']):
                            result.append((None, j, i))
                            break
            except IndexError:
                pass

    return result


def rewrite_praet(aglt_token, praet_token, by_token=None):
    """
    Copy person from aglt to praet and change praet to cond.
    """
    praet_tags = list(praet_token['tag'].split(':'))

    # praet i aglt mają tę samą liczbę
    if aglt_token is not None:
        aglt_person = aglt_token['tag'].split(':')[2]
        if aglt_token['tag'].split(':')[1] != praet_tags[1]:
            logging.warning(
                'DIFFERENT NUMBER: %s %s %s %s' % (aglt_token['tag'].split(':')[1], praet_tags[1], aglt_token, praet_token))
            return
        praet_tags.insert(3, aglt_person)

    if by_token:
        praet_tags[0] = 'cond'

    praet_token['tag'] = ':'.join(praet_tags)


def remove_tokens(sentence, aglt_indexes):
    for i in sorted(aglt_indexes, reverse=True):
        token = sentence[i]


        #dołącz do formy poprzedzającego tokenu i popraw offsety
        if token['sep']=='none':
            previous_token = sentence[i-1]
            previous_token['end']=token['end']
            previous_token['token'] += token['token']
            sentence.pop(i)

def remove_aglt(sentence, rules):
    for rule_index, rule in enumerate(rules):
        pairs = rule(sentence)

        for aglt_index, praet_index, by_index in pairs:
            if by_index is None:
                by_token = None
            else:
                by_token = sentence[by_index]

            if aglt_index is None:
                aglt_token = None
            else:
                aglt_token = sentence[aglt_index]
            rewrite_praet(aglt_token, sentence[praet_index], by_token)

        aglt_indexes = [aglt_index for aglt_index, praet_index, by_index in pairs] + [by_index for
                                                                                      aglt_index, praet_index, by_index
                                                                                      in pairs]
        aglt_indexes = [x for x in aglt_indexes if x is not None]
        remove_tokens(sentence, aglt_indexes)


def remove_aglt_from_results(results, rules):
    for paragraph in results:
        for sentence in paragraph:
            remove_aglt(sentence, rules)


def remove_aglt_from_results_rule1_3(results):
    return remove_aglt_from_results(results, [rule1, rule3])
