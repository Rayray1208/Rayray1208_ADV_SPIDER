import scrapy
from scrapy.crawler import CrawlerProcess
import os
from PIL import Image
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from fpdf import FPDF

# 爬取醫療數據與病史資料
class MedicalDataSpider(scrapy.Spider):
    name = "medical_spider"
    start_urls = ['http://example.com/medical_data']  # 替換為真實醫療網站

    def parse(self, response):
        for patient in response.css('div.patient-info'):
            patient_data = {
                'patient_id': patient.css('span.id::text').get(),
                'patient_name': patient.css('span.name::text').get(),
                'age': patient.css('span.age::text').get(),
                'gender': patient.css('span.gender::text').get(),
                'diagnosis': patient.css('span.diagnosis::text').get(),
                'medical_history': patient.css('span.history::text').get(),
                'risk_factor': patient.css('span.risk_factor::text').get(),
                'image_url': patient.css('img::attr(src)').get(),
            }
            yield patient_data

class DatabasePipeline:
    def open_spider(self, spider):
        self.engine = create_engine('sqlite:///enhanced_medical_data.db')
        self.data = []

    def close_spider(self, spider):
        df = pd.DataFrame(self.data)
        df.to_sql('enhanced_medical_records', self.engine, if_exists='replace', index=False)

    def process_item(self, item, spider):
        self.data.append(item)
        return item

# 圖像處理
def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img.save(img_path)

# 數據處理與探索 (包括可視化)
def handle_missing_data(df):
    logging.info("處理缺失數據中...")
    return df.fillna(method='ffill')

def data_statistics(df):
    logging.info("生成數據摘要和統計圖表...")
    summary = df.describe()
    sns.pairplot(df)
    plt.savefig('data_exploration.png')
    return summary

def plot_correlation(df):
    plt.figure(figsize=(10,8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("相關性矩陣")
    plt.savefig('correlation_matrix.png')

# 平滑數據處理
def smooth_data(data, method='rolling', window=5):
    if method == 'rolling':
        return data.rolling(window=window).mean()
    elif method == 'ewm':
        return data.ewm(span=window).mean()

# 增強版 RNN 模型構建
def build_model(input_shape, model_type='LSTM', units=100, dropout_rate=0.3):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units))
    elif model_type == 'GRU':
        model.add(GRU(units, return_sequences=True, input_shape=input_shape))
        model.add(GRU(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 訓練RNN模型，並返回損失趨勢
def train_rnn_model(data, epochs=200, model_type='LSTM', units=100, dropout_rate=0.3):
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = build_model((X_train.shape[1], 1), model_type=model_type, units=units, dropout_rate=dropout_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64)
    
    # 訓練完成後保存模型
    model.save(f'{model_type}_medical_risk_model.h5')
    
    return model, scaler, history

# 預測未來180天風險
def make_predictions(model, scaler, data, predict_days=180):
    scaled_data = scaler.transform(data.reshape(-1, 1))
    predictions = []
    for _ in range(predict_days):
        input_data = scaled_data[-60:].reshape(1, -1, 1)
        prediction = model.predict(input_data)
        predictions.append(prediction[0, 0])
        scaled_data = np.append(scaled_data, prediction)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 可視化模型損失
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='訓練損失')
    plt.title('模型損失趨勢')
    plt.xlabel('Epochs')
    plt.ylabel('損失')
    plt.legend()
    plt.savefig('training_loss.png')

# 生成增強版報告 (包含相關性矩陣與數據探索圖表)
def generate_report(predictions, model_type, epochs, historical_data, loss_plot_path, corr_plot_path, exploration_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # 報告標題
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="增強版醫療風險預測報告", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    # 模型資訊
    pdf.cell(200, 10, txt=f"模型類型: {model_type}", ln=True)
    pdf.cell(200, 10, txt=f"訓練次數 (Epochs): {epochs}", ln=True)

    # 未來180天風險預測
    pdf.cell(200, 10, txt="未來180天的潛在風險預測：", ln=True)
    for i, risk in enumerate(predictions):
        pdf.cell(200, 10, txt=f"第 {i+1} 天: {risk[0]:.4f}", ln=True)

    # 加入圖表 (損失、相關性、數據探索)
    pdf.image(loss_plot_path, x=10, y=100, w=100)
    pdf.add_page()
    pdf.image(corr_plot_path, x=10, y=10, w=150)
    pdf.add_page()
    pdf.image(exploration_path, x=10, y=10, w=150)

    # 訓練數據摘要
    pdf.ln(20)
    pdf.cell(200, 10, txt="歷史數據摘要：", ln=True)
    stats = historical_data.describe()
    for col in stats.columns:
        pdf.cell(200, 10, txt=f"{col}: 平均值 {stats[col]['mean']:.4f}", ln=True)

    pdf.output("enhanced_medical_risk_report.pdf")

# 主函數
if __name__ == '__main__':
    # 爬取醫療數據
    process = CrawlerProcess()
    process.crawl(MedicalDataSpider)
    process.start()

    # 圖像處理
    process_images('images')

    # 數據讀取與處理
    historical_data = pd.read_csv('long_term_medical_data.csv')
    data = historical_data['risk_metric'].values

    # 處理缺失數據與數據探索
    historical_data = handle_missing_data(historical_data)
    summary_stats = data_statistics(historical_data)
    plot_correlation(historical_data)

    # 訓練模型
    model, scaler, history = train_rnn_model(data, epochs=200, model_type='LSTM')

    # 預測未來180天
    future_risks = make_predictions(model, scaler, data)
    print("未來 180 天的風險預測：", future_risks)

    # 生成損失圖表
    plot_loss(history)

    # 生成最終報告
    generate_report(
        future_risks, 
        model_type='LSTM', 
        epochs=200, 
        historical_data=historical_data, 
        loss_plot_path='training_loss.png', 
        corr_plot_path='correlation_matrix.png', 
        exploration_path='data_exploration.png'
    )
