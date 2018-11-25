import dill
import pandas as pd

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

if __name__ == '__main__':
    l = [1, 2, 3]
    save(l, 'test')
    l_new = load('test')
    print(l_new == l)
