from datetime import datetime, timezone
import logging
import os
import re
from elasticsearch import Elasticsearch

from common_types import dict_to_markdown

# Получите URL кластера из переменной окружения
bonsai_url = os.getenv('BONSAI_URL')
access_key = os.getenv('ELASTIC_ACCESS_KEY')
access_secret_key = os.getenv('ELASTIC_SECRET_KEY')


notes_index_name="user_notes_index"

if not bonsai_url:
    raise ValueError("Переменная окружения BONSAI_URL не установлена")

es = Elasticsearch(
    [bonsai_url],
    http_auth=(access_key, access_secret_key)
)

def get_connection():
    return es

def create_indexes():
    # Настройки и маппинг 
    index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "russian_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "russian_stop",
                        "russian_stemmer"
                    ]
                }
            },
            "filter": {
                "russian_stop": {
                    "type": "stop",
                    "stopwords": "_russian_"
                },
                "russian_stemmer": {
                    "type": "stemmer",
                    "language": "russian"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "UserId": {"type": "text"},
            "Title": {
                "type": "text",
                "analyzer": "russian_analyzer"
            },
            "Body": {
                "type": "text",
                "analyzer": "russian_analyzer"
            },
            "Tags": {"type": "keyword"},
            "CreatedDate": {
                "type":   "date",
                "format": "strict_date_optional_time||epoch_millis"
                }
            }
        }
    }
    response = {}
    
    if not es.indices.exists(index=notes_index_name):
        response = es.indices.create(index=notes_index_name, body=index_settings, ignore=400)   
    else: 
        print(f'Индекс {notes_index_name} уже существует.')
        return
     #Проверка результата
    if 'acknowledged' in response:
        print(f"Индекс {notes_index_name} успешно создан.")
    else:
        print(f"Ошибка при создании индекса {notes_index_name}: {response}")

def get_elastic_datetime_now_utc():
    # Получаем текущее время в UTC
    now_utc = datetime.now(timezone.utc)
    # Преобразуем в строку в формате ISO 8601
    iso_date_utc = now_utc.replace(microsecond=0).isoformat()
    return iso_date_utc

def add_note(user_id:int, title:str, body:str, tags:list[str]):
    note_document = {
        "UserId": user_id,
        "Title": title,
        "Body": str(body),
        "Tags": tags
        }
     # Получаем текущее время в UTC
    now_utc = datetime.now(timezone.utc)
    total_seconds = int(now_utc.timestamp())
    add_or_update_document_common(index_name=notes_index_name, document=note_document, document_id=total_seconds)

def update_note(doc_id:int, user_id:int, title:str, body:str, tags:list[str]):
    note_document = {
        "UserId": user_id,
        "Title": title,
        "Body": str(body),
        "Tags": tags
        }
    add_or_update_document_common(index_name=notes_index_name, document=note_document, document_id=doc_id)


def add_or_update_document_common(index_name, document, document_id, need_to_update_documents=True):

    try:
        document['CreatedDate']=get_elastic_datetime_now_utc()
        update_body = {
            "doc": document,
            "doc_as_upsert": True
        }
        if need_to_update_documents:
            response = es.update(index=index_name, id=document_id, body=update_body)
            # print(f"Document {document_id} updated or created")
        else:
            if not es.exists(index=index_name, id=document_id):
                response = es.index(index=index_name, id=document_id, body=document)
            else:
                skipped_docs += 1
                # print(f"Document {document_id} exists and not updated")
    except Exception as e:
         logging.error(f"Error in add_or_update_document: {e}")

def get_notes_by_query(user_id:int, search_text:str, top_k=5):
    try:
        # Составление запроса
        search_query = {
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "UserId": str(user_id)
                        }
                    },
                    "must": [
                        {
                            "multi_match": {
                                "query": search_text,
                                "fields": ["Title^2", "Body", "Tags^1.5"]
                            }
                        }
                    ]
                }
            },
            "size": top_k
        }

        # Выполнение запроса
        response = es.search(index=notes_index_name, body=search_query)
        # Вывод результатов
        documents = rebuild_response(response)
        return documents
    except Exception as e:
        logging.error("Ошибка при поиске в ElasticSearch", exc_info=True)
        return []

def get_all_user_notes(user_id:int):
    try:
        # Составление запроса
        search_query = {
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "UserId": str(user_id)
                        }
                    },
                }
            },
        }

        # Выполнение запроса
        response = es.search(index=notes_index_name, body=search_query)
        # Вывод результатов
        documents = rebuild_response(response)
        return documents
    except Exception as e:
        logging.error("Ошибка при поиске в ElasticSearch", exc_info=True)
        return []
def remove_note(note_id:int):
    try:
        es.delete(index=notes_index_name, id=note_id)
        return True
    except Exception as e:
        logging.error("Ошибка при удалении в ElasticSearch", exc_info=True)
        return False

async def remove_notes(note_ids:list[int]):
    for note_id in note_ids:
        remove_note(note_id)


def rebuild_response(response):
    founded_docs=response['hits']['total']['value']
    documents = []

    logging.info(f"Общее количество найденных документов: {founded_docs}" )
    for hit in response['hits']['hits']:
        document = hit["_source"]
        document['NoteId'] = hit["_id"]
        document['Score'] = hit["_score"]
        logging.info(f"ID документа: { document['NoteId']}")
        logging.info(f"Источник: {document}")
        documents.append(document)
    return documents

    
