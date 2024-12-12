import os
import requests


weather_api_key=os.getenv('OPENWEATHERMAP_API_KEY')
weather_api_key2=os.getenv('WEATHERSTACK_API_KEY')

weather_codes = {
    0: "Ясное небо",
    1: "Преимущественно ясно",
    2: "Частично облачно",
    3: "Пасмурно",
    45: "Туман",
    48: "Туман с отложением инея",
    51: "Слабая морось",
    53: "Умеренная морось",
    55: "Сильная морось",
    56: "Слабая замерзающая морось",
    57: "Сильная замерзающая морось",
    61: "Слабый дождь",
    63: "Умеренный дождь",
    65: "Сильный дождь",
    66: "Слабый замерзающий дождь",
    67: "Сильный замерзающий дождь",
    71: "Слабый снегопад",
    73: "Умеренный снегопад",
    75: "Сильный снегопад",
    77: "Снежные зерна",
    80: "Слабый ливневый дождь",
    81: "Умеренный ливневый дождь",
    82: "Сильный ливневый дождь",
    85: "Слабые снеговые ливни",
    86: "Сильные снеговые ливни",
    95: "Гроза: слабая или умеренная",
    96: "Гроза с слабым градом",
    99: "Гроза с сильным градом"
}

# Для использования словаря можно написать функцию для получения перевода:
def get_weather_description_by_code(code):
    return weather_codes.get(code, "Неизвестный код погоды")

def get_weather_by_coordinates2(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,  # Широта
        "longitude": longitude,  # Долгота
        "current_weather": True,  # Запрос текущей погоды
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Проверка на ошибки HTTP
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе погоды: {e}")

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

def get_weather_description2(latitude, longitude):
    weather_data = get_weather_by_coordinates2(latitude, longitude)
    # Вывод результата
    if "error" in weather_data:
        result=f"Ошибка: {weather_data['error']}"
    else:
        current = weather_data.get("current_weather", {})
        weather_code = current["weathercode"]
        description = get_weather_description_by_code(weather_code)
        t= current.get("temperature", "нет данных")
        feels_like = current.get("feelslike", "нет данных")
        #pressure = current.get("pressure", "нет данных")
        #humidity = current.get("humidity", "нет данных")
        wind_speed  = current.get("windspeed", "нет данных")

        result = f"Погода: {description}, температура: {t}°C, скорость ветра {wind_speed} км/ч."
    # Возврат сообщения с описанием погоды
    return result

