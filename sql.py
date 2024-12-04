import os
import logging
import time
import mysql.connector

from common_types import SafeList

# Получение параметров подключения из переменных окружения
MYSQL_HOST = os.getenv('MYSQL_ADDON_HOST')
MYSQL_DB = os.getenv('MYSQL_ADDON_DB')
MYSQL_USER = os.getenv('MYSQL_ADDON_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_ADDON_PASSWORD')
MYSQL_PORT = os.getenv('MYSQL_ADDON_PORT', '3306')

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# администраторы
administrators_ids = []
# пользователи
user_ids= SafeList([])

def get_admins():
    return administrators_ids

async def get_all():
    return await user_ids.get_all()

#----------------------------------MySQL------------------------------------------
if not all([MYSQL_HOST, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD]):
    raise EnvironmentError("Не установлены все необходимые переменные окружения для подключения к MySQL.")

def connect_to_db():
    count = 0
    while True:
        try:
            return mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                port=MYSQL_PORT
            )
        except mysql.connector.Error as err:
            logging.error(f"Ошибка подключения к MySQL: {err}. Попытка {count} из 10.")
            count += 1
            if count > 10:
                logging.error("Подключение к MySQL не удалось.")
                raise
            else:
                # ждем 5 секунд
                time.sleep(5)
                continue


def create_tables():
    create_user_id_table()
    create_last_session_table()

def create_user_id_table():
    """Создает таблицу для хранения идентификаторов пользователей."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_ids (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id BIGINT NOT NULL
            )
        """)
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка создания таблицы в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def create_last_session_table():
    """Создает таблицу для хранения последней сессии."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS last_session (
        userid INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        last_session_time DATETIME
    )
        """)
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка создания таблицы в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

async def save_last_session(user_id, username, last_session_time):
    if user_id not in await user_ids.get_all():
        await user_ids.append(user_id)
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute(  """
                INSERT INTO last_session (userid, username, last_session_time) VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE username = VALUES(username), last_session_time = VALUES(last_session_time)
                """,
                (user_id, username, last_session_time))

        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка сохранения последней сессии в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def get_all_session():
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT userid, username, last_session_time FROM last_session")
        result = cursor.fetchall()
        return [{"userid": row[0], "username": row[1], "last_session_time": row[2]} for row in result]
    except mysql.connector.Error as err:
        logging.error(f"Ошибка при получении пользователей из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

async def save_user_id(user_id):
    if user_id not in await user_ids.get_all():
        await user_ids.append(user_id)
    """Сохраняет идентификатор пользователя в базу данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO user_ids (user_id) VALUES (%s)", (user_id,))
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка сохранения пользователя в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def get_user_ids():
    """Получает все идентификаторы пользователей из базы данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM user_ids")
        result = cursor.fetchall()
        return [row[0] for row in result]
    except mysql.connector.Error as err:
        logging.error(f"Ошибка получения пользователей из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

async def remove_user_id(user_id):
    await user_ids.remove(user_id)
    """Удаляет идентификатор пользователя из базы данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM user_ids WHERE user_id = %s", (user_id,))
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка удаления пользователя из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

#-------------------------------------end MySQL-------------------------------------------------------

def get_admins_from_os():
    global administrators_ids
    allowed_users_str = os.getenv('ALLOWED_USER_IDS', '')
    administrators_ids = [
        int(uid.strip()) for uid in allowed_users_str.split(',') if uid.strip().isdigit()
    ]
    logging.info(f"Admins uploaded from environment. Ids - {administrators_ids}")

def in_admin_list(user):
    return user.id in administrators_ids
async def in_user_list(user):
    temp_user_ids = await get_all()
    return user.id in temp_user_ids or in_admin_list(user)

#-------------------------------------end function block-------------------------------------------------------

# Получение списка администраторов из переменной окружения
get_admins_from_os() 
# Получение списка пользователей из переменной окружения
connect_to_db()
create_tables()
user_ids=SafeList(get_user_ids())