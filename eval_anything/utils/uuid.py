import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any, Union


class UUIDGenerator:
    # Class variable to store the singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, '__initialized', False):
            self.__initialized = True

    def __call__(self, to_hash: Union[dict, tuple, list, str]) -> str:
        return self.generate_uuid(to_hash)

    @classmethod
    def generate_uuid(cls, to_hash: Union[dict, tuple, list, str]) -> str:
        processed_data = cls._process_dataclasses(to_hash)
        json_string = json.dumps(processed_data, sort_keys=True)

        # Generate a hash of the JSON string
        hash_object = sha256(json_string.encode())
        hash_uid = hash_object.hexdigest()
        return hash_uid

    @classmethod
    def _process_dataclasses(cls, data: Any) -> Any:
        if is_dataclass(data):
            return cls._process_dataclasses(asdict(data))
        elif isinstance(data, dict):
            return {key: cls._process_dataclasses(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [cls._process_dataclasses(item) for item in data]
        else:
            return data
