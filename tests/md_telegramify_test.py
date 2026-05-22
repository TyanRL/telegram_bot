
import asyncio
import logging

import sys
from pathlib import Path


# 1. Определяем корень проекта (папка уровнем выше, где лежит этот файл)
project_root = Path(__file__).resolve().parent.parent

# 2. Добавляем корень в sys.path (в начало списка, чтобы ваш код имел приоритет)
sys.path.insert(0, str(project_root))

from core.md_clean import clean_message


logger = logging.getLogger(__name__)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_message = """Не совсем.

Кратко:

- **`yield`** в Python — это **генератор**:
  - возвращает значения **по одному**
  - используется в обычной итерации: `for`, `next()`
  - сам по себе **не про асинхронность**

- **`async/await`** — это **асинхронное выполнение**:
  - позволяет ждать I/O без блокировки
  - используется с `await`, `async def`
  - для итерации обычно нужен **async generator** и `async for`

### Аналогия
Оба механизма умеют **приостанавливать** выполнение функции и потом продолжать, но цель разная:

- `yield` → лениво выдавать элементы
- `await` → приостановиться, пока не завершится асинхронная операция

### Пример

**Обычный генератор:**
```python
def gen():
    yield 1
    yield 2

for x in gen():
    print(x)
```

**Асинхронная функция:**
```python
async def func():
    data = await some_io()
    return data
```

**Асинхронный генератор:**
```python
async def agen():
    yield 1
    yield 2

async for x in agen():
    print(x)
```

### Итог
Если очень грубо:  
- `yield` и `await` **похожи тем, что ставят выполнение на паузу**
- но **работают не одинаково** и решают **разные задачи**

Если хочешь, могу еще показать различие на таблице **`yield` vs `await` vs `async for`**.
"""
    
    result = asyncio.run(clean_message(test_message))
    print("=" * 40)
    print("Результат clean_measage:")
    print("=" * 40)
    print(result)
    print("=" * 40)

