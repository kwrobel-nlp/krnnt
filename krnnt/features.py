from typing import Iterable

from krnnt.utils import uniq, flatten, shape

try:
    import krnnt_utils
    from krnnt_utils import shape
except:
    pass

class FeaturePreprocessor:
    qubs = {'a', 'abo', 'aby', 'akurat', 'albo', 'ale', 'amen', 'ani', 'aż', 'aza', 'bądź', 'blisko', 'bo', 'bogać',
            'by', 'byle', 'byleby', 'choć', 'choćby', 'chociaż', 'chociażby', 'chyba', 'ci', 'co', 'coś', 'czy',
            'czyli', 'czyż', 'dalibóg', 'dobra', 'dokładnie', 'doprawdy', 'dość', 'dosyć', 'dziwna', 'dziwniejsza',
            'gdyby', 'gdzie', 'gdzieś', 'hale', 'i', 'ino', 'istotnie', 'jakby', 'jakoby', 'jednak', 'jedno', 'jeno',
            'koło', 'kontra', 'lada', 'ledwie', 'ledwo', 'li', 'maksimum', 'minimum', 'może', 'najdziwniejsza',
            'najmniej', 'najwidoczniej', 'najwyżej', 'naturalnie', 'nawzajem', 'ni', 'niby', 'nie', 'niechaj',
            'niejako', 'niejakoś', 'no', 'nuż', 'oczywiście', 'oczywista', 'okay', 'okej', 'około', 'oto', 'pewnie',
            'pewno', 'podobno', 'ponad', 'ponoś', 'prawda', 'prawie', 'przecie', 'przeszło', 'raczej', 'skąd',
            'skądinąd', 'skądże', 'szlus', 'ta', 'taj', 'tak', 'tam', 'też', 'to', 'toż', 'tuż', 'tylko', 'tylo',
            'widocznie', 'właśnie', 'wprost', 'wręcz', 'wszakże', 'wszelako', 'za', 'zaledwie', 'zaledwo', 'żali',
            'zaliż', 'zaraz', 'że', 'żeby', 'zwłaszcza'}
    safe_chars = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', u'?', u'-', u'a', u'ą', u'c', u'ć', u'b', u'e',
                  u'ę', u'd', u'g', u'f', u'i', u'h', u'k', u'j', u'm', u'l', u'ł', u'o', u'ó', u'n', u'ń', u'q', u'p',
                  u's', u'ś', u'r', u'u', u't', u'w', u'y', u'x', u'z', u'ź', u'ż'}

    @staticmethod
    def nic(form, features=None):
        return ['NIC']

    @staticmethod
    def cases(form, features=None): #TODO: wyrzucić? shape to realizuje
        if form.islower():
            return ['islower']
        elif form.isupper():
            return ['isupper']
        elif form.istitle():
            return ['istitle']
        elif form.isdigit():
            return ['isdigit']
        else:
            return ['ismixed']

    @staticmethod
    def interps(form, features):
        if 'interp' in features['tags'] and len(form) == 1:
            return [form]
        else:
            return []

    @staticmethod
    def qubliki(form, features=None):
        if form.lower() in FeaturePreprocessor.qubs:
            return [form] #TODO: form.lower()
        else:
            return []

    @staticmethod
    def shape(form, features=None):
        # print(form, shape(form))
        return [shape(form)]

    @staticmethod
    def prefix(n, form, features=None):
        try:
            char = form[n].lower()
            if char not in FeaturePreprocessor.safe_chars:
                char = '??'
        except IndexError:
            char = 'xx'

        return ['P' + str(n) + char]

    @staticmethod
    def prefix1(form, features=None):
        return FeaturePreprocessor.prefix(0, form, features)

    @staticmethod
    def prefix2(form, features=None):
        return FeaturePreprocessor.prefix(1, form, features)

    @staticmethod
    def prefix3(form, features=None):
        return FeaturePreprocessor.prefix(2, form, features)

    @staticmethod
    def suffix(n, form, features=None):
        try:
            char = form[-n].lower()
            if char not in FeaturePreprocessor.safe_chars:
                char = '??'
        except IndexError:
            char = 'xx'

        return ['S' + str(n) + char]

    @staticmethod
    def suffix1(form, features=None):
        return FeaturePreprocessor.suffix(1, form, features)

    @staticmethod
    def suffix2(form, features=None):
        return FeaturePreprocessor.suffix(2, form, features)

    @staticmethod
    def suffix3(form, features=None):
        return FeaturePreprocessor.suffix(3, form, features)


class TagsPreprocessorCython:
    @staticmethod
    def create_tags4_without_guesser(tags, features=None):
        return krnnt_utils.create_tags4_without_guesser(tags)

    @staticmethod
    def create_tags5_without_guesser(tags, features=None):
        return krnnt_utils.create_tags5_without_guesser(tags)


class TagsPreprocessor:
    cas = ['nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc']
    per = ['pri', 'sec', 'ter']
    nmb = ['sg', 'pl']
    gnd = ['m1', 'm2', 'm3', 'f', 'n']

    @staticmethod
    def create_tags4(tags, features=None, keep_guesser=True):  # concraft
        if not keep_guesser and 'ign' in tags:
            return ['ign']
            # return ['1ign','2ign','1subst:nom','2subst:sg:f','1adj:nom','1subst:gen','2subst:sg:n','2subst:sg:m1','2adj:sg:m3:pos','2subst:sg:m3','1num:acc','2num:pl:m3:rec','1brev','2adj:sg:n:pos','2num:pl:m3:congr','1num:nom','1adj:gen','1adj:loc']
        return uniq(flatten(map(lambda tag: TagsPreprocessor.create_tag4(tag), tags)))

    @staticmethod
    def create_tags4_without_guesser(tags, features=None):
        return TagsPreprocessor.create_tags4(tags, features=features, keep_guesser=False)

    @staticmethod
    def create_tag4(otag, features=None):
        tags = flatten(map(lambda x: x.split('.'), otag.split(':')))
        pos = tags[0]
        tags = tags[1:]
        tags2 = []

        first = None
        for tag in tags:
            if tag in TagsPreprocessor.cas or tag in TagsPreprocessor.per:
                first = tag
                break

        if first:
            tags.remove(first)
            tags2.append('1' + pos + ':' + first)
        else:
            tags2.append('1' + pos)  # TODO sprawdzic

        tags2.append('2' + (':'.join([pos] + tags)))

        # print otag, tags2
        return uniq(tags2)

    @staticmethod
    def create_tags5(tags, features=None, keep_guesser=True):  # concraft
        if not keep_guesser and 'ign' in tags:
            return ['ign']
            # return ['ign','sg:loc:m3','sg:nom:n','pl:nom:m3','pl:acc:m3','loc','sg:gen:m3','pl:gen:m3','sg:nom:m1','sg:nom:m3','gen','nom','acc','sg:nom:f']

        return uniq(flatten(map(lambda tag: TagsPreprocessor.create_tag5(tag), tags)))

    @staticmethod
    def create_tags5_without_guesser(tags, features=None):
        return TagsPreprocessor.create_tags5(tags, features=features, keep_guesser=False)

    @staticmethod
    def create_tag5(otag, features=None):

        tags = flatten(map(lambda x: x.split('.'), otag.split(':')))

        tags_out = []
        tags2 = []
        tags3 = []
        for tag in tags:
            if tag in TagsPreprocessor.nmb:
                tags2.append(tag)
            elif tag in TagsPreprocessor.cas:
                tags2.append(tag)
                tags3.append(tag)
            elif tag in TagsPreprocessor.gnd:
                tags2.append(tag)

        for tagsX in [tags2, tags3]:
            if tagsX:
                tags_out.append(':'.join(tagsX))

        return uniq(tags_out)

def create_token_features(token, tags, space_before): #TODO
    f = []
    f+=FeaturePreprocessor.cases(token)
    f+=FeaturePreprocessor.interps(token, {'tags':tags})
    f+=FeaturePreprocessor.qubliki(token)
    f+=FeaturePreprocessor.shape(token)  # 90%
    f+=FeaturePreprocessor.prefix1(token)
    f+=FeaturePreprocessor.prefix2(token)
    f+=FeaturePreprocessor.prefix3(token)
    f+=FeaturePreprocessor.suffix1(token)
    f+=FeaturePreprocessor.suffix2(token)
    f+=FeaturePreprocessor.suffix3(token)
    f+=TagsPreprocessorCython.create_tags4_without_guesser(
        tags)  # 3% moze cache dla wszystkich tagów
    f+=TagsPreprocessorCython.create_tags5_without_guesser(tags)  # 3%
    f+=space_before


    return f   # TODO czy uniq potrzebne - niekoenicznie: ign się powtarza