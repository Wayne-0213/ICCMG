#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import numpy as np
import sys

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk import word_tokenize

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
# from metric.meteor.meteor import Meteor
# from metric.rouge.rouge import Rouge
# from metric.cider.cider import Cider
import warnings
import argparse
import logging
import json
import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import prettytable as pt

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def calculate_meteor(sentence1, sentence2):
    """计算两个句子之间的METEOR分数"""
    # vectorizer = CountVectorizer().fit([sentence1, sentence2])
    # sentence1_vector = vectorizer.transform([sentence1])
    # sentence2_vector = vectorizer.transform([sentence2])
    # similarity = cosine_similarity(sentence1_vector, sentence2_vector)[0][0]
    # score = 2 * similarity * len(sentence1) * len(sentence2) / (len(sentence1) + len(sentence2))
    sentence1_tokens = word_tokenize(sentence1.lower())  # 对 sentence1 分词并转小写
    sentence2_tokens = word_tokenize(sentence2.lower())
    score = meteor_score([sentence1_tokens], sentence2_tokens)
    # score = meteor_score([sentence1], sentence2)
    return score

def calculate_bleu(reference, translation):
    """计算BLEU分数"""

    # reference_tokens = word_tokenize(reference.lower())  # 对 reference 分词并转小写
    # translation_tokens = word_tokenize(translation.lower())  # 对 translation 分词并转小写
    # bleu_score = sentence_bleu([reference_tokens], translation_tokens)

    bleu_score = sentence_bleu([reference], translation)
    return bleu_score

def calculate_rouge_l(reference, translation):
    """计算ROUGE-L分数"""

    # reference_tokens = word_tokenize(reference.lower())  # 对 reference 分词并转小写
    # translation_tokens = word_tokenize(translation.lower())  # 对 translation 分词并转小写
    # rouge = Rouge()
    # rouge_l_score = rouge.get_scores(" ".join(translation_tokens), " ".join(reference_tokens), avg=True)['rouge-l']

    rouge = Rouge()
    rouge_l_score = rouge.get_scores(translation, reference, avg=True)['rouge-l']
    return rouge_l_score


# def Commitbleus(refs, preds):
#
#     r_str_list = []
#     p_str_list = []
#     for r, p in zip(refs, preds):
#         if len(r[0]) == 0 or len(p) == 0:
#             continue
#         r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
#         p_str_list.append(" ".join([str(token_id) for token_id in p]))
#     try:
#         bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
#     except:
#         bleu_list = [0, 0, 0, 0]
#     codenn_bleu = bleu_list[0]
#
#     B_Norm = round(codenn_bleu, 4)
#
#     return B_Norm


# def read_to_list(filename):
#     f = open(filename, 'r',encoding="utf-8")
#     res = []
#     for row in f:
#         # (rid, text) = row.split('\t')
#         res.append(row.lower().split())
#     return res

# def metetor_rouge_cider(refs, preds):
#
#     refs_dict = {}
#     preds_dict = {}
#     for i in range(len(preds)):
#         preds_dict[i] = [" ".join(preds[i])]
#         refs_dict[i] = [" ".join(refs[i][0])]
#
#     score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
#     print("Meteor: ", round(score_Meteor*100,2))
#
#     score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
#     print("Rouge-L: ", round(score_Rouge*100,2))
#
#     score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
#     print("Cider: ",round(score_Cider,2) )



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--refs_filename', type=str, default="../saved_model/tlcodesum/UNLC/ref.txt", required=False)
    # parser.add_argument('--preds_filename', type=str, default="../saved_model/tlcodesum/UNLC/dlen500-clen30-dvoc30000-cvoc30000-bs-ddim64-cdim-rhs64-lr0_Medit_pred.txt", required=False)
    # args = parser.parse_args()

    total_meteor_score = 0
    total_bleu_score = 0
    total_rouge_l_score = 0
    num_lines = 0

    refs_filename = "saved_model/ECMG/java/test_2.gold"
    preds_filename = "saved_model/ECMG/java/test_2.output"

    # refs = read_to_list(args.refs_filename)
    with open(refs_filename, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f.readlines()[:500]]
    with open(preds_filename, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f.readlines()[:500]]
    for ref,pred in zip(refs, preds):
        num_lines += 1
        # ref=[ref]
        ref = re.sub(r'^\d+\s+', '', ref)  # 去掉行首的数字和空格
        pred = re.sub(r'^\d+\s+', '', pred)  # 同理
        print(ref)
        print(pred)
        ref = ref.strip()
        pred = pred.strip()
        try:
            score_meteor = float(calculate_meteor(ref, pred))
            bleu_score = float(calculate_bleu(ref, pred))
            rouge_l_score = float(calculate_rouge_l(ref, pred)['f'])
        except Exception as e:
            print(f"Error in line {num_lines},  error: {e}")

        # 累加总分
        total_meteor_score += score_meteor
        total_bleu_score += bleu_score
        total_rouge_l_score += rouge_l_score
    # refs = [[t] for t in refs]
    # preds = read_to_list(args.preds_filename)

    # 计算平均分
    average_meteor_score = total_meteor_score / num_lines
    average_bleu_score = total_bleu_score / num_lines
    average_rouge_l_score = total_rouge_l_score / num_lines
    print(num_lines)
    # 输出平均分
    print(f"Average METEOR Score: {average_meteor_score}")
    print(f"Average BLEU Score: {average_bleu_score}")
    print(f"Average ROUGE-L Score: {average_rouge_l_score}")




    # bleus_score = Commitbleus(refs, preds)
    # print("BLEU: %.2f"%bleus_score)
    # metetor_rouge_cider(refs, preds)


if __name__ == '__main__':
    main()
