import os
from ymaps import Geocode

# Ваш API-ключ
API_KEY = os.getenv('YMAPS_GEOCODER')

# Координаты: долгота и широта
longitude = 37.618423
latitude = 55.751244

# Создаем клиент для геокодирования
geocoder = Geocode(API_KEY)

# Выполняем обратное геокодирование
response = geocoder.reverse([longitude, latitude])

# Проверяем успешность запроса
if response and 'response' in response:
    geo_object = response['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
    address = geo_object['metaDataProperty']['GeocoderMetaData']['text']
    print(f'Адрес: {address}')
else:
    print('Не удалось получить адрес по заданным координатам.')
