import os
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        cost = time.time() - t0
        print(f"  [{func.__name__}] 耗时: {cost:.1f}s")
        return result
    return wrapper


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_pct(x: float, digits: int = 4) -> str:
    return f"{x * 100:.{digits}f}%"
