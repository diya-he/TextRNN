import operator
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors


# 创建英文词典
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

    if file == './fasttext/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" "))
                                for o in open(file, encoding='utf-8') if len(o) > 100)
    elif file == './GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
        model = KeyedVectors.load_word2vec_format(file, binary=True)
        embeddings_index = {}
        for word, vector in zip(model.vocab, model.vectors):
            embeddings_index[word] = vector
    else:
        embeddings_index = dict(get_coefs(*o.split(" "))
                                for o in open(file, encoding='latin'))

    return embeddings_index


# ## 检查预训练embeddings和vocab的覆盖情况
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}  
    nb_known_words = 0  
    nb_unknown_words = 0
    for word in tqdm(vocab):
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(
        len(known_words) / len(vocab)))  
    print('Found embeddings for  {:.2%} of all text'.format(
        nb_known_words / (nb_known_words + nb_unknown_words))) 
    unknown_words = sorted(unknown_words.items(),
                           key=operator.itemgetter(1))[::-1]
    print("unknown words : ", unknown_words[:30])
    return unknown_words



def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text



def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


# 去除英语缩写
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join(
        [mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# ## 进度条初始化
tqdm.pandas()
# ## 加载数据集
train_df = pd.read_csv("./datasets/train.csv", encoding='utf-8', sep=',')
test_df = pd.read_csv("./datasets/test.csv", encoding='utf-8', sep=',')
# test = pd.read_csv("../input/test.csv")  # Test shape = (56370, 2)
df = pd.concat([train_df, test_df])  # shape=(206916, 2)
df['Text'] = df['Text'].fillna('')

# ## 创建词典
sentences = df['Text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
# print(vocab)
# ## 加载词向量
glove = './glove/glove.6B.50d.txt'
fasttext = './fasttext/wiki-news-300d-1M.vec'
embed_glove = load_embed(glove)
embed_fasttext = load_embed(fasttext)
oov_glove = check_coverage(vocab, embed_glove)
oov_fasttext = check_coverage(vocab, embed_fasttext)

# ## 词典全部小写
print("=========转化小写后")
sentences = df['Text'].apply(lambda x: x.lower())
sentences = sentences.progress_apply(lambda x: x.split()).values
vocab_low = build_vocab(sentences)
oov_glove = check_coverage(vocab_low, embed_glove)
oov_fasttext = check_coverage(vocab_low, embed_fasttext)

# ## 去除特殊字符
print("=========去除特殊字符后")
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + \
    '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'",
                 '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
sentences = df['Text'].apply(
    lambda x: clean_special_chars(x, punct, punct_mapping))
sentences = sentences.apply(lambda x: x.lower()).progress_apply(
    lambda x: x.split()).values
vocab_punct = build_vocab(sentences)
oov_glove = check_coverage(vocab_punct, embed_glove)
oov_fasttext = check_coverage(vocab_punct, embed_fasttext)

# ## 去除’英语缩写
print("=========去除’英语缩写后")
contraction_mapping = {"here's": "here is", "it's": "it is", "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
print("- Known Contractions -")
print("   Glove :")
print(known_contractions(embed_glove))
print("   FastText :")
print(known_contractions(embed_fasttext))
sentences = df['Text'].apply(
    lambda x: clean_special_chars(x, punct, punct_mapping))
sentences = sentences.apply(lambda x: x.lower()).apply(lambda x: clean_contractions(
    x, contraction_mapping)).progress_apply(lambda x: x.split()).values
vocab_contraction = build_vocab(sentences)
oov_glove = check_coverage(vocab_contraction, embed_glove)
oov_fasttext = check_coverage(vocab_contraction, embed_fasttext)
