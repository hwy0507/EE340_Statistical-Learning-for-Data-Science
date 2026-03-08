import pandas as pd
import json
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

# 设置中文字体为simHei
plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn的字体风格
sns.set_style("darkgrid", {"font.sans-serif": ('simHei', 'arial')})

# 读取数据?
df = pd.read_csv('E:\\PythonProject\\Project2\\douban_movie.csv', encoding='utf-8')

# 读取stopword的json文件
with open('E:\\PythonProject\\stopWord.json', 'r', encoding='utf-8') as f:
    stopwords_list = json.load(f)


# 文本预处理函�?
def preprocess_text(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    tokens = jieba.cut(text)
    words = [word for word in tokens if word not in stopwords_list]
    return words


# 应用预处理函�?
df['processed_words'] = df['Comment'].apply(preprocess_text)


# 计算词频并过滤低频词
def filter_low_frequency_words(df, min_freq=3):
    all_words = []
    for words in df['processed_words']:
        all_words.extend(words)
    word_freq = Counter(all_words)

    def remove_low_freq(words):
        return [word for word in words if word_freq[word] >= min_freq]

    df['final_processed'] = df['processed_words'].apply(remove_low_freq)
    return df


# 应用低频词过�?
df = filter_low_frequency_words(df)

# 计算评论长度（字符数�?
df['comment_length'] = df['Comment'].str.len().fillna(0).astype(int)

# 计算处理后的词数
df['word_count'] = df['final_processed'].apply(len)

# 将处理后的词列表用[]框起来并保存为字符串
df['processed_text'] = df['final_processed'].apply(lambda x: f"[{', '.join(x)}]")

# 保存预处理结果到CSV
output_columns = [col for col in df.columns if col not in ['processed_words', 'final_processed']]
output_file = 'E:\\PythonProject\\Project2\\douban_movie_preprocessed.csv'
df[output_columns].to_csv(output_file, index=False, encoding='utf-8')
print(f"已将预处理结果保存至: {output_file}")

# 重新生成用于向量化的文本�?
df['processed_comment'] = df['final_processed'].apply(lambda x: " ".join(x))

# TF-IDF向量化并保存结果
def tfidf_vectorize(df):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['processed_comment'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_results = []
    for i in range(len(df)):
        feature_dict = {}
        for j in X_tfidf[i].nonzero()[1]:
            feature_dict[feature_names[j]] = round(X_tfidf[i, j], 4)
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
        tfidf_results.append("�?" + ", ".join([f"{word}:{value}" for word, value in sorted_features]) + "�?")
    return tfidf_results


# Word2Vec向量化并保存结果
def word2vec_vectorize(df):
    sentences = [text.split() for text in df['processed_comment']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    w2v_results = []
    for text in df['processed_comment']:
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            sentence_vector = np.mean(vectors, axis=0)
            vector_str = "�?" + ", ".join([f"{round(v, 4)}" for v in sentence_vector]) + "�?"
            w2v_results.append(vector_str)
        else:
            w2v_results.append("【�?")
    return w2v_results, model


# BERT向量化函数（使用GPU�?
def bert_vectorize(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = BertTokenizer.from_pretrained("./local_bert_model/")
    model = BertModel.from_pretrained("./local_bert_model").to(device)
    model.eval()

    bert_results = []
    for text in df['processed_comment']:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
            device)  # 输入数据移动到GPU

        with torch.no_grad():
            outputs = model(**inputs)
        # 取CLS token的向量（维度: [1, 768]�?
        sentence_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        vector_str = "�?" + ", ".join([f"{round(v, 4)}" for v in sentence_vector]) + "�?"
        bert_results.append(vector_str)

    return bert_results


# 执行三种向量化方�?
print("开始TF-IDF向量�?...")
df['tfidf_features'] = tfidf_vectorize(df)

print("开始Word2Vec向量�?...")
df['word2vec_features'], w2v_model = word2vec_vectorize(df)

print("开始BERT向量�?...")
df['bert_features'] = bert_vectorize(df)

# 保存所有向量化结果到CSV
vectorized_output_file = 'E:\\PythonProject\\Project2\\douban_movie_vectorized.csv'
df.to_csv(vectorized_output_file, index=False, encoding='utf-8-sig')
print(f"已将所有向量化结果保存�?: {vectorized_output_file}")

# 以下是模型训练代�?
print("\n开始模型训�?...")

# 数据准备：二分类标签
y = df['Star'].apply(lambda x: 1 if x >= 3 else 0)

# --------------------------- TF-IDF 模型 ---------------------------
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['processed_comment'])
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 逻辑回归
logistic_regression_tfidf = LogisticRegression(max_iter=1000)
logistic_regression_tfidf.fit(X_train_tfidf, y_train)
y_pred_lr_tfidf = logistic_regression_tfidf.predict(X_test_tfidf)

# 朴素贝叶�?
naive_bayes_tfidf = MultinomialNB()
naive_bayes_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = naive_bayes_tfidf.predict(X_test_tfidf)


# --------------------------- Word2Vec 模型 ---------------------------
def get_w2v_features(df, model):
    X_w2v = []
    for text in df['processed_comment']:
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            sentence_vector = np.mean(vectors, axis=0)
            X_w2v.append(sentence_vector)
        else:
            X_w2v.append(np.zeros(model.vector_size))
    return np.array(X_w2v)


X_w2v = get_w2v_features(df, w2v_model)
X_train_w2v, X_test_w2v, y_train, y_test = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

# 逻辑回归
logistic_regression_w2v = LogisticRegression(max_iter=1000)
logistic_regression_w2v.fit(X_train_w2v, y_train)
y_pred_lr_w2v = logistic_regression_w2v.predict(X_test_w2v)


# --------------------------- BERT 模型 ---------------------------
# 提取BERT特征（假设bert_features存储的是向量字符串，需要转换为数值数组）
def parse_bert_vector(s):
    s = s.strip('【�?')
    if not s:
        return np.zeros(768)  # BERT默认隐藏层维�?768
    return np.array([float(x) for x in s.split(', ')])


df['bert_vectors'] = df['bert_features'].apply(parse_bert_vector)
X_bert = np.array(df['bert_vectors'].tolist())
X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42)

# BERT + 逻辑回归
logistic_regression_bert = LogisticRegression(max_iter=1000)
logistic_regression_bert.fit(X_train_bert, y_train)
y_pred_lr_bert = logistic_regression_bert.predict(X_test_bert)

# # BERT + 朴素贝叶斯（注意：朴素贝叶斯要求特征非负，可能需要标准化�?
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_train_bert_scaled = scaler.fit_transform(X_train_bert)
# X_test_bert_scaled = scaler.transform(X_test_bert)
#
# naive_bayes_bert = MultinomialNB()  # 这里可能需要调整为GaussianNB，因为特征是连续�?
# # 或者使用StandardScaler后尝试MultinomialNB（可能效果不佳，需根据数据调整�?
# naive_bayes_bert.fit(X_train_bert_scaled, y_train)
# y_pred_nb_bert = naive_bayes_bert.predict(X_test_bert_scaled)


# --------------------------- 模型评估 ---------------------------
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{model_name} 评估结果�?")
    print(f"准确率：{accuracy:.4f}")
    print(f"精确率：{precision:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"F1值：{f1:.4f}")
    return {
        "name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# 评估所有模�?
models = [
    evaluate_model(y_test, y_pred_lr_tfidf, "TF-IDF+逻辑回归"),
    evaluate_model(y_test, y_pred_nb_tfidf, "TF-IDF+朴素贝叶�?"),
    evaluate_model(y_test, y_pred_lr_w2v, "Word2Vec+逻辑回归"),
    evaluate_model(y_test, y_pred_lr_bert, "BERT+逻辑回归"),
    #evaluate_model(y_test, y_pred_nb_bert, "BERT+朴素贝叶�?")
]


# --------------------------- 可视化对�? ---------------------------
def visualize_model_comparison(models):
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["准确�?", "精确�?", "召回�?", "F1�?"]

    x = np.arange(len(metric_names))
    width = 0.15
    multiplier = 0

    fig, ax = plt.figure(figsize=(16, 8)), plt.axes()

    for model in models:
        offset = width * multiplier
        rects = ax.bar(x + offset, [model[m] for m in metrics], width, label=model["name"])
        ax.bar_label(rects, padding=3, fmt="%.4f")
        multiplier += 1

    ax.set_ylabel("分数")
    ax.set_title("不同模型的性能对比")
    ax.set_xticks(x + width * (len(models) - 1) / 2, metric_names)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig('E:\\PythonProject\\Project2\\model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------- 其他可视化和分析（保持不变）---------------------------
def generate_wordcloud():
    # 好评词云（评�?>=4�?
    positive_words = []
    for words in df[df['Star'] >= 3]['final_processed']:
        positive_words.extend(words)
    positive_text = ' '.join(positive_words)

    # 差评词云（评�?<=2�?
    negative_words = []
    for words in df[df['Star'] <= 2]['final_processed']:
        negative_words.extend(words)
    negative_text = ' '.join(negative_words)

    # 创建词云对象
    wc = WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf',  # 确保中文字体可用
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        max_font_size=100,
        random_state=42
    )

    # 生成并保存词�?
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 好评词云
    positive_wc = wc.generate(positive_text)
    ax1.imshow(positive_wc, interpolation='bilinear')
    ax1.set_title('好评词云 (评分>=3)', fontsize=15)
    ax1.axis('off')

    # 差评词云
    negative_wc = wc.generate(negative_text)
    ax2.imshow(negative_wc, interpolation='bilinear')
    ax2.set_title('差评词云 (评分<=2)', fontsize=15)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('E:\\PythonProject\\Project2\\wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()


# 情感分布可视�?
def visualize_sentiment_distribution():
    plt.figure(figsize=(12, 6))

    # 评分分布
    plt.subplot(1, 2, 1)
    sns.countplot(x='Star', data=df)
    plt.title('评分分布')
    plt.xlabel('评分')
    plt.ylabel('评论数量')

    # 二分类情感分布（好评/差评�?
    plt.subplot(1, 2, 2)
    df['sentiment'] = df['Star'].apply(lambda x: '好评' if x >= 3 else '差评')
    sns.countplot(x='sentiment', data=df)
    plt.title('情感分布')
    plt.xlabel('情感类别')
    plt.ylabel('评论数量')

    plt.tight_layout()
    plt.savefig('E:\\PythonProject\\Project2\\sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# 相关性分�?
def analyze_correlations():
    # 计算评论长度与评分的相关�?
    length_corr = df['comment_length'].corr(df['Star'])
    word_count_corr = df['word_count'].corr(df['Star'])

    print(f"\n评论长度与评分的皮尔逊相关系�?: {length_corr:.4f}")
    print(f"评论词数与评分的皮尔逊相关系�?: {word_count_corr:.4f}")

    # 可视化相关�?
    plt.figure(figsize=(15, 6))

    # 评论长度与评分的散点�?
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Star', y='comment_length', data=df, alpha=0.5)
    plt.title('评论长度与评分的关系')
    plt.xlabel('评分')
    plt.ylabel('评论长度（字符数�?')

    # 评论词数与评分的散点�?
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Star', y='word_count', data=df, alpha=0.5)
    plt.title('评论词数与评分的关系')
    plt.xlabel('评分')
    plt.ylabel('评论词数')

    plt.tight_layout()
    plt.savefig('E:\\PythonProject\\Project2\\correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 按评分分组的评论长度统计
    length_by_star = df.groupby('Star')['comment_length'].agg(['mean', 'median', 'std']).reset_index()
    print("\n按评分分组的评论长度统计:")
    print(length_by_star)

    # 按评分分组的评论词数统计
    word_count_by_star = df.groupby('Star')['word_count'].agg(['mean', 'median', 'std']).reset_index()
    print("\n按评分分组的评论词数统计:")
    print(word_count_by_star)


# 执行所有可视化和分�?
visualize_model_comparison(models)
generate_wordcloud()
visualize_sentiment_distribution()
analyze_correlations()