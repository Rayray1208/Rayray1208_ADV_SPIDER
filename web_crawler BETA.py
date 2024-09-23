import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
import random
import time

# 模擬不同的User-Agent
ua = UserAgent()

# 使用代理列表來避免被封鎖
proxies = [
    {"http": "http://proxy1.com", "https": "https://proxy1.com"},
    {"http": "http://proxy2.com", "https": "https://proxy2.com"},
    {"http": "http://proxy3.com", "https": "https://proxy3.com"},
]

def fetch_page(url):
    try:
        # 模擬不同的請求頭
        headers = {"User-Agent": ua.random}
        # 隨機選擇一個代理
        proxy = random.choice(proxies)
        response = requests.get(url, headers=headers, proxies=proxy, timeout=10)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve {url} with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 假設我們要抓取文章標題
    title = soup.find('title').get_text()
    return title

def crawl(urls):
    results = []
    # 使用多線程處理爬蟲
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任務
        future_to_url = {executor.submit(fetch_page, url): url for url in urls}
        
        # 逐個處理已完成的任務
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                html = future.result()
                if html:
                    title = parse_html(html)
                    results.append((url, title))
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return results

if __name__ == "__main__":
    # 測試網址列表
    urls = [
        "https://www.example.com",
        "https://www.wikipedia.org",
        "https://www.python.org",
    ]
    
    start_time = time.time()
    
    results = crawl(urls)
    
    for url, title in results:
        print(f"URL: {url}, Title: {title}")
    
    print(f"爬蟲完成，總共耗時 {time.time() - start_time:.2f} 秒")
