import numpy as np
import pandas as pd

from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
from utils import get_files

import xml.etree.ElementTree as ET
from collections import Counter

# Parse the XML data
def get_text(doc_path):
    tree = ET.parse(doc_path)
    root = tree.getroot()

    # Extract metadata
    metadata = root.find('./teiHeader')
    metadata_dict = {}
    for item in metadata.findall('metadata'):
        name = item.get('name')
        value = item.text
        metadata_dict[name] = value

    # Extract text content
    text = root.find('./text')
    # body_author = text.find('./body').get('author')
    # title_author = text.find('./title').get('author')
    sentences = text.findall('body/s')
    comments = text.findall('comment')
    comments_pairs = [([(word.get('type'), word.text) for word in c.findall('s/w')], c.get('c_type')) for c in comments]
    sentences_pairs = [[(word.get('type'), word.text) for word in sent.findall('w')] for sent in sentences]
    
    tokenized_text = [word[1] for sent in sentences_pairs for word in sent]

    text = []
    for sentence in sentences_pairs:
        sentence_parsed = ''.join([word[1] for word in sentence])
        text.append(sentence_parsed)
    text = '\n'.join(text)

    c = Counter()
    for comment in comments_pairs:
        c[comment[1]] += 1
    file_name = doc_path.split('/')[-1]
    return {'date': file_name[:6], 'text': text, 'tokenized_text': tokenized_text, 'pos': c['pos'], 'neu': c['neu'], 'neg': c['neg']}

corpus = []
for file in get_files('all'):
    # sents, comments = get_text(file)
    # doc = [word[1] for sent in sents for word in sent]
    corpus.append(get_text(file))

data_df = pd.DataFrame(corpus)
data_df['num_com'] = data_df['pos'] + data_df['neu'] + data_df['neg']

print(corpus[-2])
ws_driver  = CkipWordSegmenter(model="bert-base", device=0)
# corpus_tokenized = ws_driver(corpus[:10], batch_size=128, max_length=509)
# print(corpus_tokenized)
bm25 = BM25Okapi(corpus)

def retrieve_bm25(year, month, query):
    date = year + month
    df = data_df[data_df['date']==date].reset_index(drop=True)
    corpus_tokenized = df['tokenized_text'].tolist()
    bm25 = BM25Okapi(corpus_tokenized)
    tokenized_query = ws_driver([query], batch_size=1, max_length=509)[0]
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:20]
    rel_doc = df.iloc[top_n].sort_values(by=['num_com'], ascending=False).head(5)
    return rel_doc.reset_index(drop=True)

def retrieve_tool(year, month, query):
    rel_doc = retrieve_bm25(year, month, query)
    return rel_doc['text'].tolist()[0][:1000]

if __name__ == '__main__':
    print(retrieve_bm25('2023', '02', '林智堅'))