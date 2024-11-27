
import asyncio


class SafeDict:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.data = {}

    async def set(self, key, value):
        async with self.lock:
            self.data[key] = value

    async def get(self, key):
        async with self.lock:
            return self.data.get(key)
    
    async def get(self, key, default_value):
        async with self.lock:
            return self.data.get(key, default_value)

class SafeList:
    def __init__(self, l: list):
        self.lock = asyncio.Lock()
        self.data = l

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