import operator
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors


# ## 创建英文词典
def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# ## 加载预训练词向量
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == '../@_词向量/fasttext/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file,encoding='utf-8') if len(o) > 100)
    elif file == '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
        model = KeyedVectors.load_word2vec_format(file, binary=True)
        embeddings_index = {}
        for word, vector in zip(model.vocab, model.vectors):
            embeddings_index[word] = vector
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index


# ## 检查预训练embeddings和vocab的覆盖情况
def check_coverage(vocab, embeddings_index):
    known_words = {}  # 两者都有的单词
    unknown_words = {}  # embeddings不能覆盖的单词
    nb_known_words = 0  # 对应的数量
    nb_unknown_words = 0
    #     for word in vocab.keys():
    for word in tqdm(vocab):
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))  # 覆盖单词的百分比
    print('Found embeddings for  {:.2%} of all text'.format(
        nb_known_words / (nb_known_words + nb_unknown_words)))  # 覆盖文本的百分比，与上一个指标的区别的原因在于单词在文本中是重复出现的。
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    print("unknown words : ", unknown_words[:30])
    return unknown_words


# ## 去除特殊字符
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text


# 去除英语缩写
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


if __name__ == "__main__":
    pass