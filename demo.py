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

bm25 = BM25Okapi(corpus)
ws_driver = CkipWordSegmenter(model="bert-base", device=0)
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

# def retrieve_tool(year, month, query):
#     rel_doc = retrieve_bm25(year, month, query)
#     return rel_doc['text'].tolist()[0][:1000]

def retrieve_tool(year, month, query):
    rel_doc = retrieve_bm25(year, month, query)
    return '\n'.join(rel_doc['text'].tolist())[:1000]

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

id2label = {
    0: 'negative',
    1: 'positive'
}
def sentiment_analysis(year, month, query):
    rel_doc = retrieve_bm25(year, month, query)
    doc_text = rel_doc['text'].tolist()
    x = tokenizer(doc_text, padding='longest', truncation=True, max_length=512, return_tensors="pt")
    output = model(x['input_ids'])
    sentiment = []
    for logit in output.logits:
        sentiment.append(id2label[int(logit.argmax())])
    rel_doc['senti'] = sentiment
    return sentiment 

from ckip_ner import ner_person

def ner_tool(year, month):
    date = year + month
    df = data_df[data_df['date']==date].reset_index(drop=True)
    corpus = df['text'].tolist()
    return ner_person(corpus)[0][0]

if __name__ == '__main__':
    # Import things that are needed generically
    from langchain import LLMMathChain, SerpAPIWrapper
    from langchain.agents import AgentType, initialize_agent
    from langchain.chat_models import ChatOpenAI
    from langchain.tools import BaseTool, StructuredTool, Tool, tool

    openai_api_key='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'
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    def get_input() -> str:
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)

    from langchain.tools import HumanInputRun

    tools = [
        StructuredTool.from_function(
            func=sentiment_analysis,
            name = "sentiment analysis",
            description="Useful for when you need to know the sentiment a specific month. The input of month should be a two-digit number. For example, February: 02."
        ),
        StructuredTool.from_function(
            func=ner_tool,
            name = "named entity recognition",
            description="Useful for when you need to know the most popular person of a specific month. Each input should be strings. The input of month should be a two-digit number. For example, February: 02."
        ),
        StructuredTool.from_function(
            func=retrieve_tool,
            name = "retrieval",
            description="useful for when you need to retrieve information about a specific month. Inputs shold be strings. The input of month should be a two-digit number. For example, February: 02."
        ),
        HumanInputRun(input_func=get_input)
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # prompt = '''
    # Please answer the following questions:
    # 1. Who was the most popular person in January 2022? 
    # 2. What were some retrieved information related to the person, and can you summarize it?
    # 3. What was the sentiment of said person like during that month? 
    # Repeat the previous three questions, for each of the 12 months (01, 02, 03, ..., 12) in 2022, 
    # Reply in Traditional Chinese.
    # '''

    # prompt1 = '''
    # Please answer the following questions step-by-step. After answering each question, ask a human to check whether to proceed:
    # 1. Who was the most popular person in January 2022?  
    # 2. What was the sentiment of said person like during that month? 
    # Repeat the previous two questions, for February and March 2022, 
    # Reply in Traditional Chinese.
    # '''

    prompt2 = '''
    Please answer the following questions step-by-step. After answering each question, ask a human to check whether to proceed:
    1. Who was the most popular person in January 2022?  
    2. What was the sentiment of said person like during that month? 
    Repeat the previous two questions, for February and March 2022, 
    Reply in Traditional Chinese.
    '''

    agent.run(prompt2)