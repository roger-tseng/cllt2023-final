from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# bag_list = ws_pos_bag_list

def topic_model(bag_list, topics=10):
    dictionary = Dictionary(bag_list)
    corpus = [dictionary.doc2bow(text) for text in bag_list]


    lda = LdaModel(
        corpus,
        num_topics=topics,
        id2word=dictionary,
        passes=30,
    )

    # def show_topic(topic_list, ti):
    #     topic = topic_list[ti]
    #     word_list = [word for word, prob in topic[1]]
    #     topic_string = " ".join(word_list)
    #     print(f"Topic {ti}: {topic_string}")
    #     return

    topic_list = lda.show_topics(
        num_topics=topics,
        num_words=5,
        formatted=False,
    )

    ans = ''
    for ti, topic in enumerate(topic_list):
        word_list = [word for word, prob in topic[1]]
        topic_string = " ".join(word_list)
        ans += f"Topic {ti}: {topic_string}"

    return ans

    # TODO: deal with stopwords

if __name__ == '__main__':
    pass