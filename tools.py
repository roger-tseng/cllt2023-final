import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from collections import Counter
import torch
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from rank_bm25 import BM25Okapi
import numpy as np
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


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
    body_author = text.find('./body').get('author')
    title_author = text.find('./title').get('author')
    sentences = text.findall('body/s')
    comments = text.findall('comment')
    comments_pairs = [([(word.get('type'), word.text) for word in c.findall('s/w')], c.get('c_type')) for c in comments]
    sentences_pairs = [[(word.get('type'), word.text) for word in sent.findall('w')] for sent in sentences]
    text = []
    c = Counter()
    for sentense in sentences_pairs:
        sentense_parsed = ''.join([word[1] for word in sentense])
        text.append(sentense_parsed)
    text = '\n'.join(text)
    com = []
    for comment in comments_pairs:
        c[comment[1]] += 1
        t = []
        for word in comment[0]:
            t.append(word[1])
        com.append(''.join(t))
    file_name = doc_path.split('/')[-1]
    return {'date': file_name[:6], 'text': text, 'pos': c['pos'], 'neu': c['neu'], 'neg': c['neg'], 'comment': '\n'.join(com)}


directory = '/nfs/nas-6.1/wclu/cllt/ptt_data/HatePolitics'
data = []
 
for root, dirs, files in os.walk(directory):
    for filename in tqdm(files):
        if filename != '.DS_Store':
            doc_path = os.path.join(root, filename)
            try:
                data.append(get_text(doc_path))
            except:
                continue

import pandas as pd
data_df = pd.DataFrame(data)
data_df['num_com'] = data_df['pos'] + data_df['neu'] + data_df['neg']


ner_driver = CkipNerChunker(model="bert-base", device=0)

def ner_func(doc_list):
    nlp_list = []
    
    ner = ner_driver(doc_list, batch_size=256, max_length=509)
    nlp_list.append((doc_list, None, None, ner)) 

    person_counter = Counter()
    for _, _, _, ner_doc in nlp_list:
        for sent in ner_doc:
            for token in sent:
                if token.ner == 'PERSON' and token.word != '板主群':
                    person_counter[token.word] += 1
    return person_counter

ws_driver  = CkipWordSegmenter(model="bert-base", device=0)

def bm25_retrieval_func(year, month, query):
    if year == 'all':
        df = data_df.copy()
    elif month == 'all':
        df = data_df[data_df['date'].str.contains(year)].reset_index(drop=True)
    else:
        date = year + month
        df = data_df[data_df['date']==date].reset_index(drop=True)

    corpus = df['text'].tolist()
    corpus_tokenized = ws_driver(corpus, batch_size=256, max_length=509)
    bm25 = BM25Okapi(corpus_tokenized)
    tokenized_query = ws_driver([query], batch_size=1, max_length=509)[0]
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:20]
    rel_doc = df.iloc[top_n].sort_values(by=['num_com'], ascending=False).head(5)
    return rel_doc.reset_index(drop=True)


tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment', cache_dir='/nfs/nas-6.1/wclu/cache')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment', cache_dir='/nfs/nas-6.1/wclu/cache')


id2label = {
    0: 'negative',
    1: 'positive'
}
def sentiment_tool(year, month, query):
    rel_doc = bm25_retrieval_func(year, month, query)
    com_text = rel_doc['comment'].tolist()
    x = tokenizer(com_text, padding='longest', truncation=True, max_length=512, return_tensors="pt")
    output = model(x['input_ids'])
    sentiment = []
    for logit in output.logits:
        sentiment.append(id2label[int(logit.argmax())])
    rel_doc['senti'] = sentiment
    return Counter(sentiment).most_common()[0][0]

# %%
import numpy as np
def retrieval_tool(year, month, query):
    rel_doc = bm25_retrieval_func(year, month, query)
    return '\n'.join(rel_doc['text'].tolist())[:1000]

def ner_tool(year, month):
    if year == 'all':
        df = data_df.copy()
    elif month == 'all':
        df = data_df[data_df['date'].str.contains(year)].reset_index(drop=True)
    else:
        date = year + month
        df = data_df[data_df['date']==date].reset_index(drop=True)

    corpus = df['text'].tolist()
    return ner_func(corpus).most_common()[0][0]

# %%
date_list = []
num_list = []
for date in data_df['date'].sort_values().tolist():
    if not(date in date_list):
        date_list.append(date)
for date in date_list:
    df = data_df[data_df['date']==date].reset_index(drop=True)
    corpus = df['text'].tolist()
    num_list.append(ner_func(corpus))

# %%
import matplotlib.pyplot as plt
def time_series_tool(start_date, end_date, query):
    start_index = date_list.index(start_date)
    end_index = date_list.index(end_date)
    date_interval = date_list[start_index:end_index+1]
    num_interval = num_list[start_index:end_index+1]
    pop = [ner[query] for ner in num_interval]
    fig = plt.figure(figsize=(10,3))
    lines = plt.plot(date_interval, pop) 
    plt.title( f"Mentioned times of {query}") 
    plt.xlabel("Date")
    plt.setp(lines,marker = "o")
    return 'The plot is printed out.'
