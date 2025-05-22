import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import tiktoken
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
from utils import gpt_call, num_tokens_from_messages, jaccard_similarity, text_to_split

from sentence_transformers import SentenceTransformer, util
import time
import json
import re


lan_list = ["Java"]
# prompt_template = "Generate a concise commit message that summarizes the content of code changes.\n{}code change:\n{}\ncommit message:"
# prompt_template = "Generate a commit message that summarizes the code change in this commit and describes its reason with no more than 50 words.\n{}code change:\n{}\ncommit message:"
prompt_template = "Generate a commit message for adaptive maintenance(no more than 50 words),summarizing the code change in this commit and describing its reason,which primarily involves modifications to adapt the project to a new environment,such as feature additions.\n{}code change:\n{}\ncommit message:"
# prompt_template = "Generate a commit message for corrective maintenance (no more than 50 words),summarizing the code change in this commit and describing its reason,which primarily involves fixing bugs and faults in the project.\n{}code change:\n{}\ncommit message:"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

train_id3_1 = []
with open("../dataset/Java/train_id_type3_1.txt", "r") as f:
    text = f.read().strip()
    text = text.split("##########")
    for ln in text:
        train_id3_1.append(ln)

test_id3_1 = []
with open("../dataset/Java/test_id_type3_1.txt", "r") as f:
    text = f.read().strip()
    text = text.split("##########")
    for ln in text:
        test_id3_1.append(ln)

def get_sample(lan):
    with open(f"../dataset/{lan}/test_1.json", "r") as fr:
        # data_samples = [json.loads(d) for d in fr]
        data_samples = json.load(fr)
    data_samples = data_samples[:500]

    with open(f"../dataset/{lan}/train_1.json", "r") as fr:
        # corpus = [json.loads(d) for d in fr]
        corpus = json.load(fr)
    corpus_sets = [set(text_to_split(doc["diff"])) for doc in corpus]
    print(f"corpus: {len(corpus)}")

    file_path = f"Result/{lan}_data_1_1_500.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for d in tqdm(data_samples, total=len(data_samples)):
        top_similarities = jaccard_similarity(d["diff"], corpus_sets, 10)
        top_10_indices = [index for index, _ in top_similarities]
        d["top-10"] = top_10_indices
        d["examples"] = [corpus[index] for index in top_10_indices]
        with open(file_path, 'a') as fw:
            fw.write(json.dumps(d) + '\n')



def get_sample_BM25(lan):
    with open(f"dataset/{lan}/test_1.json", "r") as fr:
        # data_samples = [json.loads(d) for d in fr]
        data_samples = json.load(fr)
    data_samples = data_samples[:200]

    with open(f"dataset/{lan}/train_1.json", "r") as fr:
        # corpus = [json.loads(d) for d in fr]
        corpus = json.load(fr)
    tokenized_corpus = [word_tokenize(entry["diff"].lower()) for entry in corpus]
    print(f"corpus: {len(corpus)}")

    file_path = f"Result/{lan}_data_1_1_bm25_ni.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for d in tqdm(data_samples, total=len(data_samples)):
        top_10_indices = compute_bm25_top_10_scores_ni(tokenized_corpus, d["diff"])
        # top_similarities = jaccard_similarity(d["diff"], tokenized_corpus, 10)
        # top_10_indices = [index for index, _ in top_similarities]
        d["top-10"] = top_10_indices.tolist()
        d["examples"] = [corpus[index] for index in top_10_indices]
        with open(file_path, 'a') as fw:
            fw.write(json.dumps(d) + '\n')

def compute_bm25_top_10_scores(tokenized_corpus, query_string):
    # Tokenize the query string
    tokenized_query = query_string.lower().split(" ")
    # Initialize BM25 with the tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)
    # Compute BM25 scores for each 'diff' entry
    doc_scores = bm25.get_scores(tokenized_query)
    # Find the indices of the entries with the top 10 BM25 scores
    top_10_indices = np.argsort(doc_scores)[::-1][:10]
    return top_10_indices

def compute_bm25_top_10_scores_ni(tokenized_corpus, query_string):
    # Tokenize the query string
    tokenized_query = query_string.lower().split(" ")
    # Initialize BM25 with the tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)
    # Compute BM25 scores for each 'diff' entry
    doc_scores = bm25.get_scores(tokenized_query)
    # Find the indices of the entries with the top 10 BM25 scores (sorted in descending order)
    top_10_indices = np.argsort(doc_scores)[::-1][:10]
    # Get the scores for the top 10 indices
    top_10_scores = doc_scores[top_10_indices]
    # Sort the top 10 indices by their BM25 scores in ascending order
    sorted_top_10_indices = top_10_indices[np.argsort(top_10_scores)]
    return sorted_top_10_indices

def get_sample_semantic(lan):
    with open(f"dataset/{lan}/test_1.json", "r") as fr:
        # data_samples = [json.loads(d) for d in fr]
        data_samples = json.load(fr)
    data_samples = data_samples[:200]

    with open(f"dataset/{lan}/train_1.json", "r") as fr:
        # corpus = [json.loads(d) for d in fr]
        corpus = json.load(fr)
    train_codes=[]
    for sample in corpus:
        train_codes.append(sample.get("diff"))
    # corpus_sets = [set(text_to_split(doc["diff"])) for doc in corpus]
    print(f"corpus: {len(corpus)}")

    file_path = f"Result/{lan}_data_1_1_semantic.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for d in tqdm(data_samples, total=len(data_samples)):
        top_10_indices = pre_process_samples_semantic(d["diff"], train_codes)
        d["top-10"] = top_10_indices
        d["examples"] = [corpus[index] for index in top_10_indices]
        with open(file_path, 'a') as fw:
            fw.write(json.dumps(d) + '\n')


def pre_process_samples_semantic(test_code, training_codes):

    # 计算单个 test_code 的 embedding
    test_code_embedding = model.encode(test_code, convert_to_tensor=True)

    training_codes_embeddings = []
    # 计算所有 training_codes 的 embeddings
    for train_code in training_codes:
        code1_emb = model.encode(train_code, convert_to_tensor=True)
        training_codes_embeddings.append(code1_emb)

    # 计算相似度得分并排序
    sim_scores = []
    for train_code_embedding in training_codes_embeddings:
        hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
        top_hit = hits[0]
        score = top_hit['score']
        sim_scores.append(score)

    # 获取得分从高到低排序的索引
    sorted_indexes = [i for i, v in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]

    # 获取前10个索引，并按照得分从低到高排序
    top_10_indexes = sorted_indexes[:10]
    sorted_top_10_indexes = sorted(top_10_indexes, key=lambda idx: sim_scores[idx])

    return sorted_top_10_indexes


def gpt_predict(sample: dict, lan: str, number: int):
    prompt = prompt_template.format(sample["example"], sample["diff"].strip())
    result = gpt_call(prompt, model="gpt-3.5-turbo-1106", temperature=0, seed=0)
    pred = result["choices"][0]["message"]["content"]
    result = {"diff_id": sample["id"], "ref": sample["message"], "pred": pred.split("\n")[0]}
    file_path = f"Result/{lan}/{number}_1_1_final_500_NoId_2.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    n=0
    for lan in lan_list:
        if not os.path.exists(f"Result/{lan}_data_1_1_500.txt"):
            get_sample(lan)
        for number in [10]:
            print(lan, number)
            with open(f"Result/{lan}_data_1_1_500.txt") as fr:
                data_samples = [json.loads(d) for d in fr.readlines()]
                i=-1
                for d in tqdm(data_samples, total=len(data_samples)):
                    i+=1
                    examples = d["examples"][-number:]
                    top_10 = d["top-10"][-number:]
                    # d["example"] = "".join([f"code change:\n{example['diff'].strip()}\n{train_id3_1[top_10[index]].strip()}\ncommit message: {example['message'].strip()}\n\n" for index, example in enumerate(examples)])
                    d["example"] = "".join([f"code change:\n{example['diff'].strip()}\ncommit message: {example['message'].strip()}\n\n" for example in examples])
                    # d["diff"] = d["diff"]+"\n"+test_id3_1[i].strip()
                    prompt = prompt_template.format(d["example"], d["diff"])
                    # print(prompt)
                    if num_tokens_from_messages(prompt) > 16384:
                        n+=1
                        used_length = num_tokens_from_messages(
                            prompt_template + "".join([f"code change:\n\ncommit message: {example['message']}\n\n" for example in examples])) #保留完整的测试示例和标识符
                        # print(used_length)
                        available_length = (16300 - used_length) // (number+1)
                        # print(available_length)
                        d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:available_length])
                        for index, example in enumerate(examples):
                            example['diff'] = encoding.decode(encoding.encode(example['diff'], disallowed_special=())[:available_length])

                        d["example"] = "".join([f"code change:\n{example['diff'].strip()}\ncommit message: {example['message'].strip()}\n\n"for example in examples])
                        prompt = prompt_template.format(d["example"], d["diff"])
                        # print(prompt)
                        # print("###################")
                    assert num_tokens_from_messages(prompt) <= 16384
        # print(n)

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                list(tqdm(executor.map(partial(gpt_predict, lan=lan, number=number), data_samples), total=len(data_samples)))
