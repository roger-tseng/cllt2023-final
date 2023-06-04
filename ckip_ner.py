from collections import Counter
import torch

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

# Initialize drivers
ws_driver  = CkipWordSegmenter(model="bert-base", device=0 if torch.cuda.is_available() else -1)
pos_driver = CkipPosTagger(model="bert-base", device=0 if torch.cuda.is_available() else -1)
ner_driver = CkipNerChunker(model="bert-base", device=0 if torch.cuda.is_available() else -1)

def ner_person(doc_list):
    nlp_list = []
    # ws  = ws_driver(doc)
    # pos = pos_driver(ws)
    ner = ner_driver(doc_list, batch_size=128)
    nlp_list.append((doc_list, None, None, ner)) 

    person_counter = Counter()
    for _, _, _, ner_doc in nlp_list:
        for sent in ner_doc:
            for token in sent:
                if token.ner == 'PERSON' and token.word != '板主群':
                    person_counter[token.word] += 1
    return person_counter.most_common(3)

if __name__ == '__main__':
    from utils import get_files
    from parse_xml import get_sentences_comments

    corpus = []
    for file in get_files('all'):
        sents, comments = get_sentences_comments(file)
        doc = [''.join([word[1] for word in sent]) for sent in sents]
        doc = '，'.join(doc)
        corpus.append(doc)
    print(corpus[9])
    print(ner_person(corpus))