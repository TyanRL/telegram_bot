import logging
import requests
from ymaps import Geocode, GeocodeAsync

from core.config import settings



async def get_address(latitude, longitude):
    try:
        # Создаем асинхронный клиент для геокодирования
        geocoder = GeocodeAsync(settings.secrets.yandex_geocoder_api_key)
    
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
        # Проверка API ключа
        if not settings.secrets.yandex_geocoder_api_key:
            logging.error("API ключ Yandex Maps не установлен или пустой")
            return None

        logging.info(f"Запрос геолокации для адреса: {address}")

        # Запрос
        url = settings.integrations.yandex_geocoder_url
        params = {
            "apikey": settings.secrets.yandex_geocoder_api_key,
            "geocode": address,
            "format": "json"
        }

        response = requests.get(
            url,
            params=params,
            timeout=settings.integrations.yandex_request_timeout_seconds,
        )

        # Обработка ответа
        if response.status_code == 200:
            try:
                data = response.json()
                logging.info(f"Ответ от Yandex API: {data}")

                # Проверяем структуру ответа
                response_data = data.get("response", {})
                geo_collection = response_data.get("GeoObjectCollection", {})
                feature_member = geo_collection.get("featureMember", [])

                if not feature_member:
                    logging.error(f"Не найдено результатов геокодирования для адреса: {address}")
                    return None

                geo_object = feature_member[0].get("GeoObject", {})
                point = geo_object.get("Point", {})
                pos = point.get("pos")

                if not pos:
                    logging.error(f"Не найдены координаты для адреса: {address}")
                    return None

                longitude, latitude = map(float, pos.split())
                logging.info(f"Получены координаты для '{address}': {latitude}, {longitude}")
                return (latitude, longitude)

            except (KeyError, IndexError, ValueError, TypeError) as e:
                logging.error(f"Ошибка при парсинге ответа API для адреса '{address}': {e}")
                logging.error(f"Структура ответа: {data}")
                return None
        else:
            logging.error(f"Ошибка HTTP {response.status_code} при запросе геолокации для '{address}': {response.text}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"Таймаут запроса геолокации для адреса: {address}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка сети при запросе геолокации для адреса '{address}': {e}")
        return None
    except Exception as e:
        logging.error(f"Неожиданная ошибка при получении геолокации по адресу '{address}': {e}", exc_info=True)
        return None
