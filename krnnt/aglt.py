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
                    result.append((i, j, None))
                    # print(sentence[i - 2:j + 2])
                    break

    return result


def rewrite_praet(aglt_token, praet_token, by_token=None):
    """
    Copy person from aglt to praet and change praet to cond.
    """
    aglt_person = aglt_token['tag'].split(':')[2]
    praet_tags = list(praet_token['tag'].split(':'))

    # praet i aglt mają tę samą liczbę
    if aglt_token['tag'].split(':')[1] != praet_tags[1]:
        logging.warning(
            'DIFFERENT NUMBER: %s %s %s %s' % (aglt_token['tag'].split(':')[1], praet_tags[1], aglt_token, praet_token))
        return

    if by_token:
        praet_tags[0] = 'cond'
    praet_tags.insert(3, aglt_person)
    praet_token['tag'] = ':'.join(praet_tags)


def remove_tokens(sentence, aglt_indexes):
    for i in sorted(aglt_indexes, reverse=True):
        sentence.pop(i)


def remove_aglt(sentence, rules):
    for rule_index, rule in enumerate(rules):
        pairs = rule(sentence)

        for aglt_index, praet_index, by_index in pairs:
            if by_index is None:
                by_token = None
            else:
                by_token = sentence[by_index]
            rewrite_praet(sentence[aglt_index], sentence[praet_index], by_token)

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
