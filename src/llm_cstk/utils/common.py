from typing import Optional, Union, Literal, Type, TypeAlias, Tuple, List, Dict, Set, Callable, Pattern


class _Singleton:
    _instance: Optional = None

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)

        return cls._instance
