import dill
import pandas as pd
import numpy as np
from sklearn import metrics

import spacy
nlp = spacy.load('en_core_web_lg') # may need to consider the large vectors model if the vectors perform well
stopwords = spacy.lang.en.STOP_WORDS

import string
punctuations = string.punctuation

PATH = '../data/pkl/'
SUB_PATH = '../data/submissions/'
TEST_PATH = '../data/test.csv'

def save(obj, obj_name):
    ''' Saves the object to a pickle file.
    
    obj: object
    Object to be pickled.

    obj_name: string
    Name of the object without the extension
    '''
    f = PATH + obj_name + '.pkl'
    dill.dump(obj, open(f, 'wb'))

def load(obj_name):
    ''' Loads an object based on name of the file.
    
    obj_name: string
    Name of the object to be loaded without the extension.
    '''
    f = PATH + obj_name + '.pkl'
    return dill.load(open(f, 'rb'))

def stack_questions(df):
    ''' Takes the pair of questions, and stacks them as individual documents to be processed.
    
    df: DataFrame 
    The data frame must have the 3 cols (id, question1, question2).
    
    return: DataFrame
    Returns a data frame of documents (questions)
    '''
    X = df.loc[:, ['id', 'question1']]
    df = df.drop(columns='question1')
    df = df.rename(columns={'question2':'question1'})
    
    X = X.append(df.loc[:, ['id', 'question1']], sort=False)
    X = X.sort_values('id').reset_index()
    
    return np.array(X['question1'])

def unstack_questions(X):
    ''' Takes X (n_questions*2, n_features) and transforms it to a (n_questions, n_features * 2) numpy array. 
    
    X: array (n_questions * 2, n_features)

    return: array (n_question, n_features*2)

    '''
    odd_idx = [i for i in range(len(X)) if i % 2 == 1]
    even_idx = [i for i in range(len(X)) if i % 2 == 0]
    
    return np.hstack([X[odd_idx], X[even_idx]])


def log_scores(model, X, y, m_name, p_cut = 0.5):
    ''' Calcuates standard classification metrics and returns a data frame to be combined with other model runs

    model: sklearn type model that implements predict_proba

    X: array (id, question1, question2)
    Array of question pairs with an id for each pair

    y: array
    Array identifying 0-1 if the pair is a duplicate or not

    m_name: string
    Name of the model to use as the index

    return: DataFrame
    DataFrame (model name, metrics). Metrics currently implemented are, 
        - accuracy
        - precision
        - recall
        - F1
        - AUC
        - Log Loss
 
    '''
    probs = model.predict_proba(X)[:, 1]
    score = (probs >= p_cut).astype(int)
    
    measures = np.array([
        metrics.accuracy_score(y, score),
        metrics.precision_score(y, score),
        metrics.recall_score(y, score),
        metrics.f1_score(y, score),
        metrics.roc_auc_score(y, probs),
        metrics.log_loss(y, probs)
    ])
    
    return pd.DataFrame(data = measures.reshape(1, -1), 
                        columns=['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss'], 
                        index=[m_name])

def generate_submissions(model, sub_name):
    ''' Generates the submission file for the competition with the provided model.

    model: sklearn type model with predict_proba implemented

    sub_name: string
    Name of the submission file
    '''
    test_df = pd.read_csv(TEST_PATH)

    # one of the test_ids is mapped to 'live in dublin?' this will be dropped
    test_df = test_df[test_df.test_id.astype(str).str.isnumeric() == True]
    test_df.loc[:, 'test_id'] = test_df.loc[:, 'test_id'].astype(int)
    
    # appears to be duplicates
    test_df = test_df.drop_duplicates()

    # some questions are blank and are flagged as na, replacing with empty string
    test_df.loc[test_df.question1.isna(), 'question1'] = ''
    test_df.loc[test_df.question2.isna(), 'question2'] = ''

    # rename test_id to id to conform to the transformation
    test_df = test_df.rename(columns={'test_id':'id'})

    probs = model.predict_proba(test_df)[:,1]

    sub_df = pd.DataFrame(columns=['test_id', 'is_duplicate'])
    sub_df.loc[:, 'test_id'] = test_df.loc[:,'id']
    sub_df.loc[:, 'is_duplicate'] = probs

    sub_df.to_csv(SUB_PATH + sub_name + '.csv', index=False)

def cleanup_text(docs):
    ''' Applies spacy lemmatization, and removes punctuation and stop words.

    docs: array-like
    Array of documents to be processed.

    retrun: array
    Array of documents with lemmatization applied.

    '''
    texts = []
    for doc in nlp.pipe(docs, disable=['parser', 'ner'], batch_size = 10000):
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    
    return np.array(texts)

def create_vectors(docs):
    ''' Converts an array of documents into spacy GloVe vectors

    docs: array
    Array of documents to be converted into vectors. This will be the average of the word vectors in the document.

    retun: array (n_docs, 300)
    Arracy of 300-d document vectors.
 
    '''
    return [doc.vector for doc in nlp.pipe(docs, disable=['parser', 'ner'])]

def ground_truth_analysis(y, y_probs):
    ''' Creates a data frame combining the ground truth with the classification model probabilities.
    
    y: array
    Ground truth array classiying the pair of questions as duplicate or not
    
    y_probs: array
    Probability of predicting the pair is a duplicate from a classifier.
    
    return: DataFrame
    DataFrame (gt, prob, diff)
        - gt: ground truth
        - prob: classifier probability
        - diff: difference between gt and prob (ascending = FP, and descending = FN)
        
    '''
    train_probs_df = pd.concat([pd.Series(y), pd.Series(y_probs)], axis=1)
    train_probs_df = train_probs_df.rename(columns={0: 'gt', 1:'prob'})
    train_probs_df['diff'] = train_probs_df.loc[:,'gt'] - train_probs_df.loc[:, 'prob']
    
    return train_probs_df

def calc_cos_sim(stack_array):
    ''' Calculates the cosine similarity between each pair of questions after a NMF reduction (or any dimension reduction)
    
    stack_array: array
    Array of vectors (n_pairs, n_dimension). Assumes pairs of questions, and thus the first half of n_dim,
    represents the first question, and the second half the other question.
    
    return: array
    Array of vectors (n_pairs, n_dimension + 1)
    
    '''
    split_idx = stack_array.shape[1] // 2
    first_q = stack_array[:, :split_idx]
    second_q = stack_array[:, split_idx:]

    sim_list = [metrics.pairwise.cosine_similarity(
                                    first_q[i].reshape(1,-1),
                                    second_q[i].reshape(1,-1)
                )[0,0]
                for i in range(stack_array.shape[0])]

    sim_list = np.array(sim_list).reshape(-1, 1)
    
    return np.hstack([stack_array, sim_list])

if __name__ == '__main__':
    l = [1, 2, 3]
    save(l, 'test')
    l_new = load('test')
    print(l_new == l)
