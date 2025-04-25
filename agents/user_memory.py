import os
import json

class UserMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_file = f"user_{user_id}_memory.json"
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as file:
                self.memory = json.load(file)
        else:
            self.memory = {}

    def save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.memory, file, indent=4)

    def update_memory(self, key: str, value: str):
        self.memory[key] = value
        self.save_memory()

    def get_memory(self, key: str) -> str:
        return self.memory.get(key, "")
