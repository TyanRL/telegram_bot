import logging
import os
import requests
from ymaps import Geocode, GeocodeAsync

# Ваш API-ключ
API_KEY = os.getenv('YMAPS_GEOCODER')


async def get_address(latitude, longitude):
    try:
        # Создаем асинхронный клиент для геокодирования
        geocoder = GeocodeAsync(API_KEY)
    
        # Выполняем обратное геокодирование
        response = await geocoder.reverse([longitude, latitude])

        # Проверяем успешность запроса
        if response and 'response' in response:
            geo_object = response['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
            address = geo_object['metaDataProperty']['GeocoderMetaData']['text']
            logging.info(f'Адрес: {address}')
            return address
        else:
            logging.error('Не удалось получить адрес по заданным координатам.')
    except Exception as e:
        logging.exception(f"Ошибка при получениии адреса: {e}")
    return None
    
def get_location_by_address(address):
    try:
        # Запрос
        url = f"https://geocode-maps.yandex.ru/1.x/"
        params = {
            "apikey": API_KEY,
            "geocode": address,
            "format": "json"
        }
        response = requests.get(url, params=params)

        # Обработка ответа
        if response.status_code == 200:
            data = response.json()
            try:
                pos = data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]["pos"]
                longitude, latitude = map(float, pos.split())
                return (latitude, longitude)
            except (IndexError, KeyError):
                return None
        else:
            logging.exception(f"Ошибка: {response.status_code}")
    except Exception as e:
        logging.exception(f"Ошибка при получениии геолокации по адресу: {e}")
    return None
