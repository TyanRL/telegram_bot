import os
import requests


weather_api_key=os.getenv('OPENWEATHERMAP_API_KEY')

def get_weather_by_coordinates(latitude, longitude):
    """
    Получает данные о погоде по широте и долготе с использованием OpenWeatherMap API.

    :param lat: Широта (latitude)
    :param lon: Долгота (longitude)
    :param api_key: Ваш API-ключ OpenWeatherMap
    :return: Словарь с данными о погоде или сообщение об ошибке
    """
    try:
        # URL для запроса к OpenWeatherMap
        url = f"https://api.openweathermap.org/data/2.5/weather"

        # Параметры запроса
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": weather_api_key,
            "units": "metric",
            "lang": "ru"
        }

        # Выполнение запроса
        response = requests.get(url, params=params)
        response.raise_for_status()  # Проверка на ошибки HTTP

        # Обработка и возврат данных
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def get_weather_description(latitude, longitude):
    weather_set = get_weather_by_coordinates(latitude, longitude)
    if "error" in weather_set:
        return weather_set["error"]

    # Получение описания погоды из ответа OpenWeatherMap
    description = weather_set.get("weather", [{}])[0].get("description")
    name = weather_set.get("name", "Not defined")
    t= weather_set.get("main", {}).get("temp", "--")
    feels_like = weather_set.get("main", {}).get("feels_like","--")
    pressure = weather_set.get("main", {}).get("pressure","--")
    humidity = weather_set.get("main", {}).get("humidity","--")
    wind_speed = weather_set.get("wind", {}).get("speed","--")
    
    result = f"""
{name}: {description}, температура: {t}°C, чувствуется как {feels_like}°C.
Давление {pressure} гПа, влажность {humidity}%, скорость ветра {wind_speed} м/с.
"""
    # Возврат сообщения с описанием погоды
    return result

