def remove_blanks_from_results(results):
    for paragraph in results:
        for sentence in paragraph:
            remove_blanks(sentence)
    return results

def remove_blanks(sentence):
    """

    """
    result = []

    i=1
    while i<len(sentence):
        token=sentence[i]
        tag = token['tag']
        if tag=='blank':
            join_token_to_previous(sentence, i)
        else:
            i+=1
    return result

def join_token_to_previous(sentence, token_id):
    previous_token = sentence[token_id - 1]
    token = sentence[token_id]

    previous_token['end'] = token['end']
    previous_token['token'] += token['token']

    #TODO: lemmas?

    sentence.pop(token_id)