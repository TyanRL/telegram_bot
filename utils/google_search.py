import logging
import os
import re
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import requests

# Ваш API-ключ и идентификатор поисковой системы (CX)
API_KEY = os.getenv('GOOGLE_API_KEY')
CX = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

def search_inner(query):
    # Создаем сервис с помощью библиотеки google-api-python-client
    service = build("customsearch", "v1", developerKey=API_KEY)
    
    # Выполняем запрос к API
    res = service.cse().list(q=query, cx=CX, num=10).execute()
    
    # Парсим результаты
    results = []
    for item in res.get("items", []):
        results.append({
            "title": item["title"],
            "link": item["link"],
            "snippet": item.get("snippet")
        })
    
    return results

async def get_search_results(query):
    try:
        results = search_inner(query)
        result_str=""
        for index, result in enumerate(results, start=1):
            result_str+=f"**{index}. {result['title']}**\n"
            result_str+=f"Link: {result['link']}\n"
            result_str+=f"Snippet: {result['snippet']}\n\n"
        return result_str, len(results)
    except Exception as e:
        logging.error(f"Ошибка при поиске в Гугл: {e}", exc_info=True)
        return None, 0
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    main_content = soup.find("div", {"id": "main-content"})
    if main_content:
        return main_content.get_text()
    else:
        return soup.get_text()  

def clean_extra_newlines_tabs(text):
    # Заменяем все табуляции \t на пробел
    text = text.replace('\t', ' ')
    
    # Удаляем лишние пробелы
    text = ' '.join(text.split())
    
    # Сохраняем одну строку между абзацами
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Заменяем два и более \n на один \n\n
    
    return text.strip()

async def get_content_by_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            pure_text=extract_text_from_html(response.content)
            pure_text=clean_extra_newlines_tabs(pure_text)
            return pure_text  # Возвращает содержимое страницы
        else:
             logging.error(f"Ошибка загрузки страницы: {response.status_code}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Ошибка при получении контента по ссылке {url}: {e}", exc_info=True)


