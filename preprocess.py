import re
from nltk.tokenize import word_tokenize
import datetime

def print_log(log,new_line=False):
    now = datetime.datetime.now()
    if new_line == False:
        print('['+str(now)+']'+' - ',log)
    else:
        print('['+str(now)+']')
        print(log)

def process_data(data, lancaster, en_stops,doc_name):
    print_log('======================= PRE-PROCESSING '+doc_name+' ========================')

    # STEP 1 - Convert string to lowercase
    cleaned_data = data.lower()
    print_log('Changing doc to lower case ...')
    # STEP 2 - Remove all non alphabetical characters
    print_log('Removing non alphabetical characters ...')
    regex = re.compile('[^a-zA-Z ]')
    cleaned_data = regex.sub('', cleaned_data)
    # STEP 3 - Tokenizing words
    print_log('Tokenizing ...')
    cleaned_data_tokenized = word_tokenize(cleaned_data)
    # STEP 4 - Removing stop words
    print_log('Removing stop words ...')
    for index, word in enumerate(cleaned_data_tokenized):  # iterating on a copy since removing will mess things up
        if word in en_stops:
            cleaned_data_tokenized.remove(word)
        else:
            # STEP 5 - Stemming words
            cleaned_data_tokenized[index] = lancaster.stem(word)
    print_log('======================= PRE-PROCESSING COMPLETE '+doc_name+' =================')
    return ' '.join(cleaned_data_tokenized)