import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import jieba
import json
import re
from collections import Counter

# --------------------------- 读取原始数据 ---------------------------
df = pd.read_csv('E:\\PythonProject\\Project2\\douban_movie.csv', encoding='utf-8')

# 读取停用词（如果需要）
with open('E:\\PythonProject\\stopWord.json', 'r', encoding='utf-8') as f:
    stopwords_list = json.load(f)


# --------------------------- 重新执行必要的预处理 ---------------------------
# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    tokens = jieba.cut(text)
    words = [word for word in tokens if word not in stopwords_list]
    return words


# 应用预处理
df['processed_words'] = df['Comment'].apply(preprocess_text)


# 过滤低频词
def filter_low_frequency_words(df, min_freq=3):
    all_words = []
    for words in df['processed_words']:
        all_words.extend(words)
    word_freq = Counter(all_words)

    def remove_low_freq(words):
        return [word for word in words if word_freq[word] >= min_freq]

    df['final_processed'] = df['processed_words'].apply(remove_low_freq)
    return df


df = filter_low_frequency_words(df)

# 生成 processed_comment 列
df['processed_comment'] = df['final_processed'].apply(lambda x: " ".join(x))

# --------------------------- 数据准备（1%样本）---------------------------
df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
y = df['Star'].apply(lambda x: 1 if x >= 3 else 0)


# --------------------------- BERT向量化函数 ---------------------------
def bert_vectorize_small_sample(df, device):
    tokenizer = BertTokenizer.from_pretrained("./local_bert_model/")
    model = BertModel.from_pretrained("./local_bert_model").to(device)
    model.eval()

    bert_results = []
    for text in df['processed_comment']:
        if not text.strip():
            bert_results.append(np.zeros(768))
            continue

        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        bert_results.append(cls_vector)

    return np.array(bert_results)


# --------------------------- 执行BERT向量化 ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
X_bert = bert_vectorize_small_sample(df, device)

# --------------------------- 模型训练与评估 ---------------------------
X_train_bert, X_test_bert, y_train, y_test = train_test_split(
    X_bert, y, test_size=0.2, random_state=42
)
model_bert = LogisticRegression(max_iter=500)
model_bert.fit(X_train_bert, y_train)
y_pred_bert = model_bert.predict(X_test_bert)
accuracy = accuracy_score(y_test, y_pred_bert)
print(f"\nBERT+逻辑回归 准确率（1%样本）: {accuracy:.4f}")