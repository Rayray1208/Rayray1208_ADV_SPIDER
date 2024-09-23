import scrapy
from scrapy.crawler import CrawlerProcess
import os
from PIL import Image
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import tensorflow as tf
import tensorboard 
from keras.models import Sequential, save_model
from keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from fpdf import FPDF
from sklearn.metrics import mean_squared_error

# 設定日誌紀錄
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

class MedicalDataSpider(scrapy.Spider):
    name = 'medical_data'
    start_urls = ['https://example.com/medical-database']  # 目標網站
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 16,
        'RETRY_TIMES': 3,
        'ITEM_PIPELINES': {
            '__main__.DatabasePipeline': 1,
        }
    }

    def parse(self, response):
        patient_urls = response.xpath('//a[@class="patient-link"]/@href').getall()
        for patient_url in patient_urls:
            yield response.follow(patient_url, self.parse_patient)

        next_page = response.xpath('//a[@class="next"]/@href').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_patient(self, response):
        patient_data = {
            'name': response.xpath('//h1/text()').get(),
            'age': response.xpath('//div[@class="age"]/text()').get(),
            'diagnosis': response.xpath('//div[@class="diagnosis"]/text()').get(),
            'history': response.xpath('//div[@class="history"]/text()').get(),
            'image_urls': response.xpath('//img/@src').getall()
        }

        for image_url in patient_data['image_urls']:
            yield response.follow(image_url, self.download_image, meta={'patient_name': patient_data['name']})

        yield patient_data

    def download_image(self, response):
        patient_name = response.meta['patient_name']
        if not os.path.exists(f'images/{patient_name}'):
            os.makedirs(f'images/{patient_name}')
        image_name = response.url.split('/')[-1]
        with open(f'images/{patient_name}/{image_name}', 'wb') as f:
            f.write(response.body)

class DatabasePipeline:
    def __init__(self):
        self.engine = create_engine('sqlite:///medical_data.db')

    def process_item(self, item, spider):
        pd.DataFrame([item]).to_sql('patients', self.engine, if_exists='append', index=False)
        return item

def process_images(image_folder):
    processed_folder = os.path.join(image_folder, 'processed')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png', '.gif', '.jpeg')):
            img = Image.open(os.path.join(image_folder, filename))
            img = img.resize((256, 256))
            img.save(os.path.join(processed_folder, filename))

def handle_missing_data(data):
    return data.fillna(method='ffill')

def smooth_data(data, method='rolling', window_size=3):
    if method == 'rolling':
        return data.rolling(window=window_size).mean().dropna()
    elif method == 'exponential':
        return data.ewm(span=window_size).mean().dropna()
    return data

def prepare_time_series_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(model_type='LSTM', input_shape=(30, 1)):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    else:
        raise ValueError("Unsupported model type: Choose 'LSTM' or 'GRU'")
    
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_rnn_model(data, epochs=200, model_type='LSTM', units=100, dropout_rate=0.3):
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < 60:
        raise ValueError("數據集太小，無法生成足夠的訓練樣本")
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 檢查數據是否為空
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if X_train.shape[0] == 0:
        raise ValueError("X_train 是空的，請檢查數據處理流程")

    # 確保 X_train 的形狀正確
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = build_model((X_train.shape[1], 1), model_type=model_type, units=units, dropout_rate=dropout_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64)
    
    # 訓練完成後保存模型
    model.save(f'{model_type}_medical_risk_model.h5')
    
    return model, scaler, history

    # 顯示訓練過程的損失圖
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('training_loss.png')
    
def train_rnn_model(data, time_step=30, epochs=50, model_type='LSTM'):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    scaled_data = handle_missing_data(pd.Series(scaled_data.flatten()))
    scaled_data = smooth_data(scaled_data)

    X, y = prepare_time_series_data(scaled_data.values, time_step)
    print(f"X.shape: {X.shape}")
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model(model_type=model_type, input_shape=(X.shape[1], 1))

    # 使用 Early Stopping 來防止過擬合
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    history = model.fit(X, y, batch_size=1, epochs=epochs, verbose=1, callbacks=[callback])

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('training_loss.png')

    save_model(model, 'medical_risk_model.h5')

    logging.info(f"{datetime.now()} - Model trained with {epochs} epochs using {model_type}.")
    
    return model, scaler

def make_predictions(model, scaler, data, time_step=30, future_steps=180):
    last_data = data[-time_step:].reshape(1, time_step, 1)
    predictions = []
    
    for _ in range(future_steps):
        prediction = model.predict(last_data)
        predictions.append(prediction[0, 0])
        last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def generate_report(future_risks, model_type, epochs, historical_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 添加標題
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Medical Risk Prediction Report', 0, 1, 'C')

    # 添加模型信息
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, f'Model Type: {model_type}', 0, 1)
    pdf.cell(0, 10, f'Epochs: {epochs}', 0, 1)

    # 添加歷史數據摘要
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Historical Data Summary:', 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, historical_data.describe().to_string())

    # 添加預測結果
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, 'Future Risk Predictions (Next 180 Days):', 0, 1)

    for day, risk in enumerate(future_risks.flatten(), start=1):
        pdf.cell(0, 10, f'Day {day}: Risk = {risk:.4f}', 0, 1)

    # 添加損失變化圖
    pdf.add_page()
    pdf.image('training_loss.png', x=10, y=30, w=190)

    # 添加訓練紀錄
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Training Log', 0, 1)
    pdf.set_font("Arial", '', 12)

    with open('training_log.txt', 'r') as log_file:
        for line in log_file:
            pdf.multi_cell(0, 10, line.strip())

    pdf_file = 'medical_risk_report.pdf'
    pdf.output(pdf_file)
    print(f"Report generated: {pdf_file}")

if __name__ == '__main__':
    # 爬蟲部分
    process = CrawlerProcess()
    process.crawl(MedicalDataSpider)
    process.start()

    process_images('images')

    # 假設這裡有一個數據檔案，包含長時間的醫療數據
    historical_data = pd.read_csv('long_term_medical_data.csv')  # 長期數據
    data = historical_data['risk_metric'].values  # 假設是風險指標

    # 訓練模型
    model, scaler = train_rnn_model(data)

    # 做出未來的預測
    future_risks = make_predictions(model, scaler, data)
    print("未來 180 天的潛在風險預測：", future_risks)

    # 生成報告
    generate_report(future_risks, 'LSTM', 50, historical_data)
