# Thanks to https://github.com/PavelOstyakov/toxic
import sys
import numpy as np
import pandas as pd
import nltk
import tqdm


sentence_length = 256


def tokenize_sentences(sentences, word_to_token):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        words = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in words:
            word = word.lower()
            if word not in word_to_token:
                word_to_token[word] = len(word_to_token)
            token = word_to_token[word]
            result.append(token)
        tokenized_sentences.append(result)

    return tokenized_sentences, word_to_token


def load_embedding_list(file_path, word_to_token):
    word_to_embed = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm.tqdm(f.readlines()[1:-1]):
            data = row.split(' ')
            word = data[0]
            if word in word_to_token:
                embedding = np.array([float(num) for num in data[1:-1]],
                                     dtype=np.float32)
                embedding_list.append(embedding)
                word_to_embed[word] = len(word_to_embed)

    return embedding_list, word_to_embed


def token_to_embedding(tokenized_sentences, token_to_word,
                       word_to_embed, sentence_length):
    words_train = []
    for sentence in tokenized_sentences:
        current_words = []
        for token in sentence:
            word = token_to_word[token]
            embed_idx = word_to_embed.get(word, -1)
            # skip unknown words
            if embed_idx != -1:
                current_words.append(embed_idx)
                if len(current_words) == sentence_length:
                    break

        # padding to same length
        end_idx = len(word_to_embed)
        if len(current_words) < sentence_length:
            current_words += [end_idx] * (sentence_length - len(current_words))
        words_train.append(current_words)
    return words_train


train_csv = 'dataset/train.csv'
if len(sys.argv) > 1:
    train_csv = sys.argv[1]
print('Train file: {}'.format(train_csv))

print('Tokenizing train set...')
df = pd.read_csv(train_csv)
list_sentences = df['comment_text'].fillna('').values
tokenized_sentences_train, word_to_token = tokenize_sentences(list_sentences,
                                                              {})

print('Tokenizing test set...')
df = pd.read_csv('dataset/test.csv')
list_sentences = df['comment_text'].fillna('').values
tokenized_sentences_test, word_to_token = tokenize_sentences(list_sentences,
                                                             word_to_token)

print('Loading embeddings...')
embedding_list, word_to_embed = load_embedding_list('dataset/word.vec',
                                                    word_to_token)
embedding_size = len(embedding_list[0])
embedding_list.append(np.zeros(embedding_size, dtype=np.float32))   # end

token_to_word = {token: word for word, token in word_to_token.items()}
del word_to_token

print('Coverting embeddings...')
train_embedding = token_to_embedding(
    tokenized_sentences_train,
    token_to_word,
    word_to_embed,
    sentence_length)

test_embedding = token_to_embedding(
    tokenized_sentences_test,
    token_to_word,
    word_to_embed,
    sentence_length)

np.savez('dataset/text-embedding.npz', train_embedding=train_embedding,
         test_embedding=test_embedding, embedding_list=embedding_list)
np.savez('dataset/valid-words.npz', valid_words=set(token_to_word.values()))
