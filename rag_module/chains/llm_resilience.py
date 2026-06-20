"""
Retry com backoff e timeout para chamadas ao LLM.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("forzy.llm_resilience")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator factory para retry com exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "[LLM] Tentativa %d/%d falhou: %s",
                            attempt + 1,
                            max_retries,
                            exc,
                        )
                        logger.warning(
                            "[LLM] Aguardando %.1fs antes de tentar novamente",
                            delay,
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def with_timeout(seconds: float = 15.0):
    """Decorator que lança TimeoutError se a função exceder o tempo limite."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as exc:
                    error[0] = exc

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                raise TimeoutError(f"LLM não respondeu em {seconds}s")
            if error[0]:
                raise error[0]
            return result[0]

        return wrapper

    return decorator


def resilient_llm_call(
    func: Callable,
    fallback_func: Callable | None = None,
    max_retries: int = 3,
    timeout: float = 15.0,
) -> Any:
    """Executa func com timeout e retry; usa fallback se todas as tentativas falharem."""

    @retry_with_backoff(max_retries=max_retries, base_delay=2.0)
    @with_timeout(seconds=timeout)
    def _call():
        return func()

    try:
        return _call()
    except Exception:
        if fallback_func is not None:
            logger.warning(
                "[LLM] Todas as tentativas falharam. Usando fallback local."
            )
            return fallback_func()
        raise
