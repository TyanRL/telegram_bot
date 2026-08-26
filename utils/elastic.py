from datetime import datetime, timezone
import logging
import re
from elasticsearch import Elasticsearch

from core.common_types import dict_to_markdown
from core.config import settings

notes_index_name = settings.elastic.notes_index_name
es: Elasticsearch | None = None

def get_connection():
    global es
    if es is None:
        if not settings.elastic.url:
            raise EnvironmentError("Не задан elastic.url в config.yaml")
        es = Elasticsearch(
            [settings.elastic.url],
            http_auth=(
                settings.secrets.elastic_access_key,
                settings.secrets.elastic_secret_key,
            ),
        )
    return es

def create_indexes():
    client = get_connection()
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
    
    if not client.indices.exists(index=notes_index_name):
        response = client.indices.create(
            index=notes_index_name,
            body=index_settings,
        )
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
            response = get_connection().update(index=index_name, id=document_id, body=update_body)
            # print(f"Document {document_id} updated or created")
        else:
            if not get_connection().exists(index=index_name, id=document_id):
                response = get_connection().index(index=index_name, id=document_id, body=document)
            else:
                # print(f"Document {document_id} exists and not updated")
                pass
    except Exception as e:
         logging.error(f"Error in add_or_update_document: {e}")

def get_notes_by_query(
    user_id: int,
    search_text: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_k: int | None = None,
):
    try:
        must_clauses = []

        # Полнотекстовый поиск или match_all
        if search_text == "*":
            # Если хотите вернуть все документы, удовлетворяющие условиям фильтра, используем match_all.
            must_clauses.append({"match_all": {}})
        elif search_text:
            must_clauses.append({
                "multi_match": {
                    "query": search_text,
                    "fields": ["Title^2", "Body", "Tags^1.5"]
                }
            })

        # Диапазонный фильтр по дате
        if start_date or end_date:
            date_range = {}
            if start_date:
                date_range["gte"] = start_date
            if end_date:
                date_range["lte"] = end_date

            must_clauses.append({
                "range": {
                    "CreatedDate": date_range
                }
            })

        search_query = {
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "UserId": str(user_id)
                        }
                    },
                    "must": must_clauses
                }
            },
            "size": top_k if top_k is not None else settings.elastic.search_top_k,
        }

        response = get_connection().search(index=notes_index_name, body=search_query)
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
        response = get_connection().search(index=notes_index_name, body=search_query)
        # Вывод результатов
        documents = rebuild_response(response)
        return documents
    except Exception as e:
        logging.error("Ошибка при поиске в ElasticSearch", exc_info=True)
        return []
def remove_note(note_id:int):
    try:
        get_connection().delete(index=notes_index_name, id=str(note_id))
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

    
