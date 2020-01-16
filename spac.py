import spacy
from spacy import displacy
sp = spacy.load('en_core_web_sm')

sen = sp(u'Can you google it?')
word = sen[2]
print(word)
print(f'{word.text:{5}} {word.pos_:{5}} {word.tag_:{5}} {spacy.explain(word.tag_)}')

sen = sp(u"I like to play football. I hated it in my childhood though")

num_pos = sen.count_by(spacy.attrs.POS)
print(num_pos)
for k,v in sorted(num_pos.items()):
    print(f'{k}. {sen.vocab[k].text:{8}}: {v}')

sen = sp(u"I like to play football. I hated it in my childhood though")
displacy.render(sen, style='dep', jupyter=True, options={'distance': 85})
displacy.serve(sen, style='dep', options={'distance': 120})