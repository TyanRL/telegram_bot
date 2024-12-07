import logging
import os
from ymaps import Geocode

# Ваш API-ключ
API_KEY = os.getenv('YMAPS_GEOCODER')

# Координаты: долгота и широта
longitude = 37.618423
latitude = 55.751244

# Создаем клиент для геокодирования
geocoder = Geocode(API_KEY)

async def get_address(longitude, latitude):
    # Выполняем обратное геокодирование
    response = await geocoder.reverse([longitude, latitude])

    # Проверяем успешность запроса
    if response and 'response' in response:
        geo_object = response['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
        address = geo_object['metaDataProperty']['GeocoderMetaData']['text']
        logging.info(f'Адрес: {address}')
    else:
        logging.error('Не удалось получить адрес по заданным координатам.')
