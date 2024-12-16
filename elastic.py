import os
import re
from elasticsearch import Elasticsearch

# Получите URL кластера из переменной окружения
bonsai_url = os.getenv('BONSAI_URL')
access_key = os.getenv('ELASTIC_ACCESS_KEY')
access_secret_key = os.getenv('ELASTIC_SECRET_KEY')

if not bonsai_url:
    raise ValueError("Переменная окружения BONSAI_URL не установлена")

es = Elasticsearch(
    [bonsai_url],
    http_auth=(access_key, access_secret_key)
)

def get_connection():
    return es
