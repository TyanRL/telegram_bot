import logging
import time
import mysql.connector

from core.common_types import SafeList
from core.config import settings

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.application.log_level.upper(), logging.INFO)
)

# администраторы
administrators_ids = []
# пользователи
user_ids= SafeList([])

def get_admins():
    return administrators_ids

async def get_all():
    return await user_ids.get_all()

#----------------------------------MySQL------------------------------------------

def connect_to_db():
    count = 0
    while True:
        try:
            return mysql.connector.connect(
                host=settings.mysql.host,
                user=settings.mysql.user,
                password=settings.secrets.mysql_password,
                database=settings.mysql.database,
                port=settings.mysql.port,
            )
        except mysql.connector.Error as err:
            logging.error(
                "Ошибка подключения к MySQL: %s. Попытка %s из %s.",
                err,
                count,
                settings.mysql.connection_retries,
            )
            count += 1
            if count > settings.mysql.connection_retries:
                logging.error("Подключение к MySQL не удалось.")
                raise
            else:
                time.sleep(settings.mysql.retry_delay_seconds)
                continue


def create_tables():
    create_user_id_table()
    create_last_session_table()

def create_user_id_table():
    """Создает таблицу для хранения идентификаторов пользователей."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {settings.mysql.user_ids_table_name} (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
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



async def save_user_id(user_id):
    if user_id not in await user_ids.get_all():
        await user_ids.append(user_id)
    """Сохраняет идентификатор пользователя в базу данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"INSERT INTO {settings.mysql.user_ids_table_name} (user_id) VALUES (%s)",
            (user_id,),
        )
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
        cursor.execute(f"SELECT DISTINCT user_id FROM {settings.mysql.user_ids_table_name}")
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
        cursor.execute(
            f"DELETE FROM {settings.mysql.user_ids_table_name} WHERE user_id = %s",
            (user_id,),
        )
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка удаления пользователя из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def create_last_session_table():
    """Создает таблицу для хранения последней сессии."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {settings.mysql.last_session_table_name} (
        userid BIGINT AUTO_INCREMENT PRIMARY KEY,
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

    if username is None or username == '':
        username = 'NDU'

    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute(  f"""
                INSERT INTO {settings.mysql.last_session_table_name} (userid, username, last_session_time) VALUES (%s, %s, %s)
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
        cursor.execute(f"""
        SELECT 
            userid, 
            username, 
            last_session_time 
        FROM
            {settings.mysql.last_session_table_name}
        ORDER BY 
            last_session_time DESC 
        LIMIT 10;
        """)

        result = cursor.fetchall()
        return [{"userid": row[0], "username": row[1], "last_session_time": row[2]} for row in result]
    except mysql.connector.Error as err:
        logging.error(f"Ошибка при получении пользователей из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()



#-------------------------------------end MySQL-------------------------------------------------------

def get_admins_from_os():
    global administrators_ids
    administrators_ids = list(settings.access.allowed_user_ids)
    logging.info(f"Admins loaded from config. Ids - {administrators_ids}")

def in_admin_list(user):
    return user.id in administrators_ids
async def in_user_list(user):
    temp_user_ids = await get_all()
    return user.id in temp_user_ids or in_admin_list(user)

#-------------------------------------end function block-------------------------------------------------------

def init_db():
    global user_ids
    if not all(
        [
            settings.mysql.host,
            settings.mysql.database,
            settings.mysql.user,
            settings.secrets.mysql_password,
        ]
    ):
        return
        raise EnvironmentError(
            "Не заданы все необходимые настройки для подключения к MySQL."
        )
    get_admins_from_os()
    connect_to_db()
    create_tables()
    user_ids = SafeList(get_user_ids())
