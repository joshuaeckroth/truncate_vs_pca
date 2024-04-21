#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tiktoken
import os
import pandas as pd
from time import sleep
from pathlib import Path
from openai import OpenAI, RateLimitError


# In[2]:


Path("output").mkdir(exist_ok=True)


# In[3]:


df_train = pd.read_parquet('data/train.parquet')
df_train.sample(5)


# In[4]:


df_test = pd.read_parquet('data/test.parquet')
df_test.sample(5)


# In[5]:


text_emb_3_large_cost = 0.13 / 1000000  # 0.13 USD per 1M tokens
ENCODING = tiktoken.get_encoding("cl100k_base")
emb_column = "text"
encoded_text_train = ENCODING.encode(df_train[emb_column].str.cat(sep=" "))
encoded_text_test = ENCODING.encode(df_test[emb_column].str.cat(sep=" "))
print(f"This embedding operation will cost a total of {(len(encoded_text_train) + len(encoded_text_test)) * text_emb_3_large_cost:.2f} USD.")


# In[6]:


key = os.environ.get("API_KEY")
client = OpenAI(api_key=key)


# In[7]:


def get_embeddings_batch(text, model="text-embedding-3-large"):
    batch_embeds = client.embeddings.create(input=text, model=model)
    embeds = [e.embedding for e in batch_embeds.data]
    return embeds


# In[8]:


def add_embeddings(df):
    batch_size = 512
    embedding_batches = []
    embeddings = [item for sublist in embedding_batches for item in sublist]
    current_batch = len(embeddings) // batch_size + 1
    total_batches = len(df) // batch_size + 1
    for i in range(current_batch, total_batches + 1):
        start_ind = (i - 1) * batch_size
        end_ind = min(start_ind + batch_size - 1, len(df))
        batch = df.loc[start_ind:end_ind, emb_column].tolist()
        print(f"Processing records for batch {i} from {start_ind} to {end_ind}")
        for j in range(10):
            try:
                embedding_batches.append(get_embeddings_batch(batch))
                break
            except RateLimitError:
                print(f"Rate limit error, waiting 15 seconds and trying again (attempt {j + 1})")
                sleep(15)
    embeddings = [item for sublist in embedding_batches for item in sublist]
    df["emb_large"] = embeddings
    return df


# In[9]:


df_train_emb = add_embeddings(df_train)
df_train_emb.to_parquet("output/train_emb.parquet")


# In[10]:


df_test_emb = add_embeddings(df_test)
df_test_emb.to_parquet("output/test_emb.parquet")

