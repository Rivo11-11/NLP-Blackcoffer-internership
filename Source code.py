#!/usr/bin/env python
# coding: utf-8

# ## 1) Data Extraction

# In[2]:


## importing file .. converting xlsx to csv for compatibility issues
import pandas as pd

# Step 1: Read URLs from CSV
df = pd.read_csv('Input.csv')
df.head(10)


# In[3]:


## We have a 114 urls
urls = df['URL'].tolist()
urls_ids = df["URL_ID"].tolist()




# In[4]:


print(urls[0])


# In[26]:


import requests
from bs4 import BeautifulSoup
import os

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "DNT": "1",
    "Connection": "close",
    "Upgrade-Insecure-Requests": "1"
}

# Create subfolder if it doesn't exist
if not os.path.exists('articles'):
    os.makedirs('articles')

for i, url in enumerate(urls):
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find the article title
    title_element = soup.find('h1', class_='entry-title')
    if title_element:
        article_title = title_element.text.strip()
    else:
        article_title = "Title not found"

    # Find the article content
    content_element = soup.find('div', class_='td-post-content tagdiv-type')
    if content_element:
        article_content = content_element.text.strip()
    else:
        article_content = "Content not found"
    file_name = f'articles/{urls_ids[i]}.txt'

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(f'Title: {article_title}\n\n')
        file.write(article_content)

    print(f'Article saved as {file_name}')






# In[5]:


import os
subfolder_path = "articles"
files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
text_files = [f for f in files if f.endswith(".txt")]
num_text_files = len(text_files)
print(f"Number of text files in 'articles': {num_text_files}")


# ## 2) Data Analysis

# ### ***Sentimental Analysis***

# In[6]:


import os
import nltk
from nltk.tokenize import word_tokenize

# Step 1: Load Stop Words Lists and Master Dictionary
# Load the Stop Words Lists and Master Dictionary here

def load_positive_negative_words(positive_words_path, negative_words_path):
    positive_words = set()
    negative_words = set()

    # Load positive words
    with open(positive_words_path, 'r', encoding='utf-8') as file:
        positive_words.update(word.strip() for word in file.readlines() if word.strip().upper() not in stop_words)

    # Load negative words
    with open(negative_words_path, 'r', encoding='utf-8') as file:
        negative_words.update(word.strip() for word in file.readlines() if word.strip().upper() not in stop_words )

    return positive_words, negative_words

def load_stop_words(folder_path):
    stop_words = set()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r', encoding='latin-1') as file:
                words = file.read().split()
                stop_words.update(word.upper() for word in words)
    return stop_words
# Function to clean text using stop words
def clean_text(text, stop_words):
    words = word_tokenize(text)
    cleaned_words = [word for word in words if  word.upper() not in stop_words]
    return ' '.join(cleaned_words)

# Function to perform sentiment analysis
def perform_sentiment_analysis(text, positive_words, negative_words):
    positive_score = sum(1 for word in text.split() if word.lower() in positive_words)
    negative_score = sum(1 for word in text.split() if word.lower() in negative_words)
    
    # Polarity Score Calculation
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    
    # Subjectivity Score Calculation
    total_words = len(text.split())
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score



# ### ***Analysis of Readability and others***

# In[7]:


import syllables
import re

def average_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return len(words) / len(sentences)

def count_complex_words(text, stop_words):
    words = nltk.word_tokenize(text)
    complex_words = [word for word in words if syllables.estimate(word) > 2 and word.upper() not in stop_words]
    return len(complex_words)
def count_total_cleaned_words(text, stop_words):
    words = nltk.word_tokenize(text)
    cleaned_words = [word for word in words if word.upper() not in stop_words and word.isalpha()]
    return len(cleaned_words)
def count_syllables_in_word(word):
    vowels = "aeiouy"
    count = 0
    prev_char = 'z'

    for char in word.lower():
        if char in vowels and prev_char not in vowels:
            count += 1
        prev_char = char

    if word.lower().endswith(("es", "ed")) and count > 1:
        count -= 1

    return max(count, 1)

# Function to count total syllables in a text
def count_syllables_in_text(text):
    words = nltk.word_tokenize(text)
    syllable_count = sum(count_syllables_in_word(word) for word in words)
    return syllable_count
def count_personal_pronouns(text):
    personal_pronouns = re.findall(r'\b(?:I|we|my|ours|us)\b', text, flags=re.IGNORECASE)
    return len(personal_pronouns)

# Function to calculate average word length
def average_word_length(text):
    words = nltk.word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    return float(total_characters)/ len(words)


# In[8]:


import nltk
nltk.download('cmudict')
nltk.download('punkt')  


# ### ***Call Function***

# In[9]:


# Initialize the result Lists
results = []

# Define paths to Article Lists Stop Words Lists and Master Dictionary
subfolder_path = "articles"
stop_words_path = "StopWords"
master_dict_path = "MasterDictionary"
positive_words_path = "MasterDictionary/positive-words.txt"
negative_words_path = "MasterDictionary/negative-words.txt"

# Load Stop Words Lists and Master Dictionary
stop_words = load_stop_words(stop_words_path)
positive_words, negative_words = load_positive_negative_words(positive_words_path, negative_words_path)
# Load positive and negative words from Master Dictionary (after excluding stop words)

# Iterate through the text files articles
for file_name in os.listdir(subfolder_path):
    if file_name.endswith(".txt"):
        with open(os.path.join(subfolder_path, file_name), 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Step 2: Clean Text using Stop Words
        cleaned_content = clean_text(content, stop_words)
        
        # Step 3: Perform Sentiment Analysis
        positive_score, negative_score, polarity_score, subjectivity_score = perform_sentiment_analysis(cleaned_content, positive_words, negative_words)
        avg_sentence_length = average_sentence_length(cleaned_content)
        percentage_complex_words = count_complex_words(cleaned_content, stop_words) / len(cleaned_content.split())
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        complex_word_count = count_complex_words(cleaned_content,stop_words)
        avg_words_per_sentence = average_sentence_length(cleaned_content)
        word_count = count_total_cleaned_words(cleaned_content,stop_words)
        Syllable_Count_Per_Word = count_syllables_in_text(cleaned_content)
        Count_personel_pronouns = count_personal_pronouns(content)
        averag_word_length = average_word_length(cleaned_content)
        ## Append to the result
        parts = file_name.rsplit('.', 1)
        file_id = parts[0]

        result_row = [file_id, positive_score, negative_score,polarity_score,subjectivity_score,avg_sentence_length,percentage_complex_words,fog_index,avg_words_per_sentence,complex_word_count,word_count,Syllable_Count_Per_Word,Count_personel_pronouns,averag_word_length]  # Add computed variables here
        results.append(result_row)


# In[10]:


# column_names = ['file_id', 'positive_score', 'negative_score' , 'polarity_score', 'subjectivity_score','Avg sentence words','Percentage of complex words','Fog index']
column_names = ['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE' , 'POLARITY SCORE', 'SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']
df_results = pd.DataFrame(results, columns=column_names)
df_results


# In[17]:


filtered_df = df_results[df_results["URL_ID"] == 123.0]
filtered_df


# In[18]:


filtered_df = df_results[df_results["URL_ID"] == 4321.0]
filtered_df


# In[15]:


# Assuming df1 and df2 are your DataFrames
df['URL_ID'] = df['URL_ID'].astype('float')
df_results['URL_ID'] = df_results['URL_ID'].astype('float')
merged_df = pd.merge(df, df_results, on='URL_ID', how='inner')
merged_df.head(10)


# In[19]:


## Finally Export 
merged_df.to_csv('Output Data Structure.csv', index=False)


# In[ ]:




