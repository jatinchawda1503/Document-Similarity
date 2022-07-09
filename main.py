from preprocess import process_data
from preprocess import print_log
from difflib import SequenceMatcher
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def print_line_space():
    print(' ')
    print(' ')
    print(' ')

def check_similarity():
    documents = {}
    files_read = ''
    for file in os.listdir('files'):
        filename = os.fsdecode(file)
        files_read = files_read + filename + ','
        with open('files/'+filename) as f:
            documents[filename] = f.read()
    print_log('Files read and being checked for plagarism -'+files_read[:-1])
    print_line_space()
    cleaned_data = pre_process_data(documents)
    print_line_space()

    print_log('======================= CHECKING DOCUMENT SIMILARITY USING TFIDF ========================')
    check_similarity_tfidf(cleaned_data,files_read[:-1])
    print_line_space()

    print_log('======================= CHECKING DOCUMENT SIMILARITY USING DIFFLIB ========================')
    check_similarity_difflib(cleaned_data,files_read[:-1])
    print_line_space()


def pre_process_data(documents):
    lancaster = LancasterStemmer()
    en_stops = set(stopwords.words('english'))
    cleaned_data = []
    for name, text in documents.items():
        processed_data = process_data(text, lancaster, en_stops, name)
        cleaned_data.append(processed_data)
    return cleaned_data

def check_similarity_tfidf(cleaned_data,files_read):

    docs_tfid = TfidfVectorizer().fit_transform(cleaned_data)
    similarity = docs_tfid * docs_tfid.T
    print_log('======================= TFIDF SIMILARITY SCORES =======================================')
    scores_dataframe = get_similarity_dataframe(similarity.toarray(),files_read.split(','))
    print_log(scores_dataframe,True)
    print_log('======================= END OF TFIDF ========================')
    print_log('=============================================================')
    print_log('=============================================================')
    print_log('=============================================================')
    print_log('=============================================================')

def check_similarity_difflib(cleaned_data,files_read):
    scores_data = []
    row_data = []
    for i in range(0,len(cleaned_data)):
        if len(row_data) > 0:
            scores_data.append(row_data)
        row_data = []
        for j in range(0,len(cleaned_data)):
            if i == j:
                row_data.append(1)
            else:
                similarity = SequenceMatcher(None, cleaned_data[i], cleaned_data[j]).ratio()
                row_data.append(similarity)
    if len(row_data) > 0:
        scores_data.append(row_data)
    print_log('======================= DIFFLIB SEQUENCE MATCHER SIMILARITY SCORES ========================')
    scores_dataframe = get_similarity_dataframe(scores_data, files_read.split(','))
    print_log(scores_dataframe, True)
    print_log('======================= END OF DIFFLIB ========================')
    print_log('=============================================================')
    print_log('=============================================================')
    print_log('=============================================================')
    print_log('=============================================================')

def get_similarity_dataframe(list,column_names):
    df = pd.DataFrame(list,columns=column_names, index=column_names)
    return df

print_log('======================= SCRIPT EXECUTION STARTED =======================================')
check_similarity()