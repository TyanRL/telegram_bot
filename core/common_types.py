
from dataclasses import dataclass, field
import asyncio


@dataclass
class ModelAnswer:
    bot_reply: str | None = None
    additional_system_messages: list[dict] = field(default_factory=list)
    ctx_token: int = 0
    completion_token: int = 0
    recurse: bool = False


class SafeDict:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.data = {}

    async def set(self, key, value):
        async with self.lock:
            self.data[key] = value

    async def get(self, key, default_value):
        async with self.lock:
            return self.data.get(key, default_value)
    
    async def delete(self, key):
        async with self.lock:
            if key in self.data:
                del self.data[key]

class SafeList:
    def __init__(self, list: list):
        self.lock = asyncio.Lock()
        self.data = list

    async def append(self, value):
        async with self.lock:
            self.data.append(value)

    async def get(self, index):
        async with self.lock:
            return self.data[index]
        
    async def remove(self, value):
        async with self.lock:
            if value in self.data:
                self.data.remove(value)
    
    async def get_all(self):
        async with self.lock:
            return list(self.data)
        

def dict_to_markdown(d, indent=0):
    result = []
    for key, value in d.items():
        indentation = '  ' * indent
        if isinstance(value, dict):
            result.append(f"{indentation}## {key}")
            result.append(dict_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            result.append(f"{indentation}### {key}")
            for item in value:
                result.append(f"{indentation}- {item}")
        else:
            result.append(f"{indentation}- **{key}**: {value}")
    return "\n".join(result)