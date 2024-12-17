from flask import Flask, request, render_template, jsonify
import tensorflow as tf
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

def manual_pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:  # 截断
            if truncating == 'pre':
                new_seq = seq[-maxlen:]
            else:  # 'post'
                new_seq = seq[:maxlen]
        else:  # 填充
            if padding == 'pre':
                new_seq = [value] * (maxlen - len(seq)) + seq
            else:  # 'post'
                new_seq = seq + [value] * (maxlen - len(seq))
        padded_sequences.append(new_seq)
    return np.array(padded_sequences)


model = tf.keras.models.load_model('fake_news_model.h5', custom_objects={})


with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

max_len = 450



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 获取用户输入的新闻内容
        news_text = request.form['news']
        sequences = tokenizer.texts_to_sequences([news_text])
        
        # 手动实现填充
        padded_sequences = np.zeros((1, max_len))
        seq = sequences[0]
        padded_sequences[0, :len(seq)] = seq[:max_len]
        
        # 模型预测
        prediction = model.predict(padded_sequences)
        result = "True News" if prediction[0][0] > 0.5 else "Fake News"
        confidence = round(float(prediction[0][0]) * 100, 2) if result == "True News" else round((1 - float(prediction[0][0])) * 100, 2)

        # 渲染主页并显示结果
        return render_template(
            'index.html',
            news=news_text,
            prediction=result,
        )
    else:
        # 初次访问时渲染空页面
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port = 5000)
