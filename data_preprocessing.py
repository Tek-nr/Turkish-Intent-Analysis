import re
import pandas as pd

def data_cleaning(text):
    text = text.replace("\n", " ").lower() # Replace line breaks with space and convert the text to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    text = re.sub('RT', '', text)
    text = re.sub('[^a-zA-ZğüşıöçĞÜŞİÖÇ]', ' ', text)
    return text

def remove_stopwords(text, stopwords):
    # Splitted the text by space, filter out stopwords, and join the cleaned words
    return " ".join(word for word in text.split() if word.lower() not in stopwords)

def word_tokenize(text):
    words = re.findall(r'\b\w+\b', text)  # Used a regular expression to split the text into words
    return words

def balance_data(df_patterns):
    df_intent = df_patterns['intent']
    max_counts = df_intent.value_counts().max() #max number of examples for a class
    
    new_df = df_patterns.copy()
    for i in df_intent.unique():
        i_count = int(df_intent[df_intent == i].value_counts())
        if i_count < max_counts:
            i_samples = df_patterns[df_intent == i].sample(max_counts - i_count, replace = True, ignore_index = True)
            new_df = pd.concat([new_df, i_samples])
    return new_df