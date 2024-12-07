import logging
import os
from ymaps import Geocode, GeocodeAsync

# Ваш API-ключ
API_KEY = os.getenv('YMAPS_GEOCODER')


async def get_address(latitude, longitude):
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
        return None
