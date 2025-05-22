import json
import re

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# nltk.download('wordnet')
# 文件路径
input_file = "Result/Java/10_1_1_final_500_semantic_4.txt"  # 输入文件
output_file = "output.txt"  # 输出文件

# 评分计算函数
def calculate_meteor(sentence1, sentence2):
    """计算两个句子之间的METEOR分数"""
    # vectorizer = CountVectorizer().fit([sentence1, sentence2])
    # sentence1_vector = vectorizer.transform([sentence1])
    # sentence2_vector = vectorizer.transform([sentence2])
    # similarity = cosine_similarity(sentence1_vector, sentence2_vector)[0][0]
    # score = 2 * similarity * len(sentence1) * len(sentence2) / (len(sentence1) + len(sentence2))
    sentence1_tokens = word_tokenize(sentence1.lower())  # 对 sentence1 分词并转小写
    sentence2_tokens = word_tokenize(sentence2.lower())
    sentence2_tokens = " ".join(sentence2_tokens)
    # print(type(sentence1_tokens))
    # print(type(sentence2_tokens))
    # print(sentence1_tokens)
    # print(sentence2_tokens)
    score = meteor_score(sentence1_tokens, sentence2_tokens)

    # score = meteor_score([sentence1],sentence2)
    return score

def calculate_bleu(reference, translation):
    """计算BLEU分数"""
    # sentence1_tokens = word_tokenize(reference.lower())  # 对 sentence1 分词并转小写
    # sentence2_tokens = word_tokenize(translation.lower())
    # sentence2_tokens = " ".join(sentence2_tokens)
    bleu_score = sentence_bleu([reference], translation)
    # bleu_score = sentence_bleu(sentence1_tokens, sentence2_tokens)
    return bleu_score

def calculate_rouge_l(reference, translation):
    """计算ROUGE-L分数"""
    rouge = Rouge()
    rouge_l_score = rouge.get_scores(translation, reference, avg=True)['rouge-l']
    return rouge_l_score

def is_camel_case(s):
    """判断是否为驼峰命名"""
    return s != s.lower() and s != s.upper() and "_" not in s

def to_underline(x):
    """将驼峰命名转换为下划线形式"""
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', ' \g<0>', x).lower()

# 初始化变量
total_meteor_score = 0
total_bleu_score = 0
total_rouge_l_score = 0
num_lines = 0

# 打开输入文件，读取前 200 行
with open(input_file, 'r') as f:
    lines = f.readlines()[:500]  #   200 行

# 处理每一行
for line in lines:
    num_lines += 1
    stripped_line = line.strip()
    if stripped_line:
        try:
            data = json.loads(stripped_line)
            # 处理 data
        except json.JSONDecodeError:
            print("Invalid JSON in line, skipping:", num_lines)
            continue  # 跳过无效行
    diff_id = data['diff_id']
    ref = data['ref']

    # if ref is None or ref == "":
    #     print(f"Error in line {num_lines}, diff_id: {diff_id}, ref is empty")
    # words = ref.split()
    # ref_list = []
    # for word in words:
    #     if len(word) > 1:
    #         if is_camel_case(word):
    #             ref_list.append(to_underline(word))
    #         else:
    #             ref_list.append(word)
    #     else:
    #         ref_list.append(word)
    #
    # ref = ' '.join(ref_list)

    pred = data['pred']

    # wordsGPT = pred.split()
    # pred_list = []
    # for wordGPT in wordsGPT:
    #     if len(wordGPT) > 1:
    #         if is_camel_case(wordGPT):
    #             pred_list.append(to_underline(wordGPT))
    #         else:
    #             pred_list.append(wordGPT)
    #     else:
    #         pred_list.append(wordGPT)
    # pred = ' '.join(pred_list)
    # print(pred)
    # 对 ref 和 pred 进行驼峰命名转换
    # ref_words = ref.split()
    # pred_words = pred.split()
    # ref = ' '.join([to_underline(word) if is_camel_case(word) else word for word in ref_words])
    # pred = ' '.join([to_underline(word) if is_camel_case(word) else word for word in pred_words])

    # 计算各项分数
    try:
        score_meteor = float(calculate_meteor(ref, pred))
        bleu_score = float(calculate_bleu(ref, pred))
        rouge_l_score = float(calculate_rouge_l(ref, pred)['f'])
    except Exception as e:
        print(f"Error in line {num_lines}, diff_id: {diff_id}, error: {e}")

    # 累加总分
    total_meteor_score += score_meteor
    total_bleu_score += bleu_score
    total_rouge_l_score += rouge_l_score

    # 保存结果
    result = {
        "diff_id": diff_id,
        "ref": ref,
        "pred": pred,
        "METEOR Score": score_meteor,
        "BLEU Score": bleu_score,
        "ROUGE-L Score": rouge_l_score
    }
    # 写入输出文件
    with open(output_file, 'a') as out_file:
        json.dump(result, out_file)
        out_file.write('\n')

# 计算平均分
average_meteor_score = total_meteor_score / num_lines
average_bleu_score = total_bleu_score / num_lines
average_rouge_l_score = total_rouge_l_score / num_lines
print(num_lines)
# 输出平均分
print(f"Average METEOR Score: {average_meteor_score}")
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Average ROUGE-L Score: {average_rouge_l_score}")
