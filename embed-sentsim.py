import tiktoken
import os
import pandas as pd
import sys
from time import sleep
from pathlib import Path
from openai import OpenAI, RateLimitError
import datasets

Path("output").mkdir(exist_ok=True)

ds = datasets.load_dataset("embedding-data/sentence-compression")["train"]
df = pd.DataFrame(ds)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

# reduce size
df_train = df_train.sample(5000, random_state=42)
df_test = df_test.sample(5000, random_state=42)

df_train.to_parquet("output/train_noemb.parquet")
df_test.to_parquet("output/test_noemb.parquet")

print(df_train.sample(5))
print("Number of training rows:", len(df_train))
print(df_test.sample(5))
print("Number of testing rows:", len(df_test))

text_emb_3_large_cost = 0.13 / 1000000  # 0.13 USD per 1M tokens
ENCODING = tiktoken.get_encoding("cl100k_base")
total_tokens = df_train["set"].apply(lambda x: len(ENCODING.encode(x[0])) + len(ENCODING.encode(x[1]))).sum() + \
        df_test["set"].apply(lambda x: len(ENCODING.encode(x[0])) + len(ENCODING.encode(x[1]))).sum()
print(f"Total number of tokens in the dataset: {total_tokens}")
print(f"This embedding operation will cost a total of {total_tokens * text_emb_3_large_cost:.2f} USD.")

key = os.environ.get("API_KEY")
client = OpenAI(api_key=key)

def get_embeddings_batch(text, model="text-embedding-3-large"):
    batch_embeds = client.embeddings.create(input=text, model=model)
    embeds = [e.embedding for e in batch_embeds.data]
    return embeds

def add_embeddings(df):
    batch_size = 512
    embedding_batches = []
    embeddings = []
    total_batches = len(df) // batch_size + 1
    for i in range(total_batches + 1):
        start_ind = i * batch_size
        end_ind = min(start_ind + batch_size, len(df))
        batch = df.iloc[start_ind:end_ind]["set"].tolist()
        if len(batch) == 0:
            break
        print(f"Processing records for batch {i+1} from {start_ind} to {end_ind-1}")
        for j in range(10):
            try:
                firsts = [x[0] for x in batch]
                seconds = [x[1] for x in batch]
                firsts_embeds = get_embeddings_batch(firsts)
                seconds_embeds = get_embeddings_batch(seconds)
                embedding_batches.append(list(zip(firsts_embeds, seconds_embeds)))
                break
            except RateLimitError:
                print(f"Rate limit error, waiting 15 seconds and trying again (attempt {j + 1})")
                sleep(15)
    embeddings = [item for sublist in embedding_batches for item in sublist]
    df["emb_large"] = embeddings
    return df

df_train_emb = add_embeddings(df_train)
df_train_emb.to_parquet("output/train_emb.parquet")

df_test_emb = add_embeddings(df_test)
df_test_emb.to_parquet("output/test_emb.parquet")

