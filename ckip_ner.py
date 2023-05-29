from collections import Counter
import torch

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

# Initialize drivers
ws_driver  = CkipWordSegmenter(model="bert-base", device=0 if torch.cuda.is_available() else -1)
pos_driver = CkipPosTagger(model="bert-base", device=0 if torch.cuda.is_available() else -1)
ner_driver = CkipNerChunker(model="bert-base", device=0 if torch.cuda.is_available() else -1)

def ner_person(doc_list):
    nlp_list = []
    for doc in doc_list:
        # ws  = ws_driver(doc)
        # pos = pos_driver(ws)
        ner = ner_driver(doc)
        nlp_list.append((doc, None, None, ner)) 

    person_counter = Counter()
    for _, _, _, ner_doc in nlp_list:
        for sent in ner_doc:
            for token in sent:
                if token.ner == 'PERSON' and token.word != '板主群':
                    person_counter[token.word] += 1