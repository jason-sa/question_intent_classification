import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logging.info('Loading libraries...')

import numpy as np
import pandas as pd 
from nltk import ngrams
import utils
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


logging.info('Loading data...')

## load data
utils.PATH = 'data/'
cleaned_questions = utils.load('cleaned_questions')
clean_tokens = utils.load('clean_tokens')
clean_question_features = utils.load('clean_question_features')
lemma_tokens = utils.load('lemma_tokens')
lemma_question_features = utils.load('lemma_question_features')
qa_df = utils.load('qa_df')

# logging.info(f'cleaned_questions shape: {len(cleaned_questions)}')
# logging.info(f'clean_tokens shape: {len(clean_tokens)}')
# logging.info(f'clean_question_features shape: {clean_question_features.shape}')
# logging.info(f'lemma_tokens shape: {len(lemma_tokens)}')
# logging.info(f'lemma_question_features shape: {lemma_question_features.shape}')
# logging.info(f'qa_df shape: {qa_df.shape}')

logging.info('Loading models...')

## load best fitted model
xgb = utils.load('xgb_FINAL_model_question_swapped')

# build features on the cleaned text only
clean_text_features = Pipeline(
    [
        ('clean', FunctionTransformer(utils.clean_questions, validate=False)),
        ('dist', FunctionTransformer(utils.add_min_max_avg_distance_features, validate=False))
    ]
)

# build features on the cleanned and lemmatized text features
lemma_text_features = Pipeline(
    [
        ('clean', FunctionTransformer(utils.clean_questions, validate=False)),
        ('lemma', FunctionTransformer(utils.apply_lemma, validate=False)),
        ('dist', FunctionTransformer(utils.add_min_max_avg_distance_features, validate=False))
    ]
)


def ngram_similarity(q_token, token_db,  n_grams=[1, 2, 3]):
    ''' Calculates the ngram similarity between a pair of questions. Similarity is defined as,
            2 · ( |S1| / |S1 ∩ S2| + |S2| / |S1 ∩ S2|)^−1
        where S_i is the ngrams for question i
        
        n_grams: list
        List of n-grams to calculate, i.e. [1, 2, 3]
        
        return: array-like (n_pairs, len(n_grams))
        N-dimensional array of n_gram similarity calculated for the different n_grams.
        
    '''
    ngram_sim = []
    ngram_q2 = [set(ngrams(q_token, i, pad_right=True)) for i in n_grams]
    for t in token_db:
        ngram_q1 = [set(ngrams(t, i, pad_right=True)) for i in n_grams]

        doc_ngram_sim = []
        for i in range(len(ngram_q1)):
            try:
                s1 = len(ngram_q1[i]) / len(ngram_q1[i].intersection(ngram_q2[i]))
            except:
                s1 = 0

            try:
                s2 = len(ngram_q2[i]) / len(ngram_q1[i].intersection(ngram_q2[i]))
            except:
                s2 = 0

            if s1 == 0 and s2 == 0:
                doc_ngram_sim.append(0)
            else:
                doc_ngram_sim.append(2 * (s1 + s2)**-1)
        ngram_sim.append(doc_ngram_sim)
        
    return np.array(ngram_sim)

def ask_question(question, n):
    
    logging.info(f'Processing question: {question}')
    question_clean = utils.clean_questions([question]) ## returns an array

    logging.info('Building feature set 1')
    ## Feature Set 1 -- clean text similarity
    ## create the tokens for the question
    doc = utils.nlp(question_clean[0])
    question_tokens = doc.to_array([utils.spacy.attrs.LOWER])
    clean_n_gram = ngram_similarity(question_tokens, clean_tokens)

    # logging.info(f'Clean shape: {clean_n_gram.shape}')

    logging.info('Building feature set 2')
    ## Feature Set 2 -- Clean distance features
    # union single question features
    question_features = clean_text_features.transform([question]) 
    clean_single_features = np.hstack([clean_question_features, 
                                        np.repeat(question_features, clean_question_features.shape[0], axis=0)])

    # logging.info(f'Clean shape: {clean_single_features.shape}')
    
    logging.info('Building feature set 3')
    ## Feature Set 3 -- Lemma text similarity
    # calculate n_gram similarity for the cleaned and lemmatized question
    question_lemma = utils.apply_lemma(question_clean)
    doc = utils.nlp(question_lemma[0])
    question_tokens = doc.to_array([utils.spacy.attrs.LOWER])
    lemma_n_gram = ngram_similarity(question_tokens, lemma_tokens)

    # logging.info(f'Clean shape: {lemma_n_gram.shape}')

    logging.info('Building feature set 4')
    ## Feature Set 4 -- Lemma distance features
    # union single question features
    question_features = lemma_text_features.transform([question]) 
    lemma_single_features = np.hstack([lemma_question_features, 
                                        np.repeat(question_features, lemma_question_features.shape[0], axis=0)])
    
    # logging.info(f'Lemma shape: {lemma_single_features.shape}')
    
    logging.info('Making prediction')
    # combine the entire features space
    feature_space = np.hstack([ clean_n_gram, 
                                clean_single_features, 
                                lemma_n_gram, 
                                lemma_single_features])

    # make the prediction
    probs = xgb.predict_proba(feature_space)[:, 1]

    top = probs.argsort()[-n:][::-1]
    top_question = np.array(qa_df.iloc[top]).reshape(len(top), -1)
    top_probs = probs[top].reshape(len(top), 1)

    return zip(top_question, top_probs)

if __name__ == '__main__':
    print('Welcome to the q&a bot!\nPlease enter your question. Once you are done enter "exit"')
    q = input('Question: ')
    if q != 'exit':
        a = input('How many answers? ')

    while q != 'exit':
        results = ask_question(q, int(a))
        print('\n\nResults:\n')
        for a, p in results:
            print(a, p)
            print()

        print('\n\n')
        q = input('Question: ')
        if q != 'exit':
            a = input('How many answers? ')

