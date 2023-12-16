import numpy as np
import pickle as pkl
from gensim.models import KeyedVectors


# ## 加载预训练词向量
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == './fasttext/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8') if len(o) > 100)
    elif file == './GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
        model = KeyedVectors.load_word2vec_format(file, binary=True)
        embeddings_index = {}
        for word, vector in zip(model.vocab, model.vectors):
            embeddings_index[word] = vector
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index


ebed = []

def get_embed(vocab_path, embed_path,dim):
    vocab = pkl.load(open(vocab_path, 'rb'))
    embed_glove = load_embed(embed_path)
    for v in vocab:
        if v not in embed_glove.keys():
            ebed.append(np.asarray([0 for i in range(0,dim)], dtype='float32'))
        else:
            ebed.append(embed_glove[v])
    return np.asarray(ebed, dtype='float32')


vocab_path = './datasets/vocab.pkl'
embed_path = './glove/glove.6B.300d.txt'
dim = 300

np.savez('./datasets/glove.6B.300d.npz',embeddings=get_embed(vocab_path, embed_path, dim))
