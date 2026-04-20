"""na_memorix 插件日志辅助。"""

from __future__ import annotations

import logging
import re
from typing import Any

_FALLBACK_LOGGER_PREFIX = "na_memorix"
_PLUGIN_INSTANCE: Any | None = None


def _format_component_name(raw_name: str) -> str:
    """把模块 logger 名称归一化为插件内组件名。"""

    candidate = str(raw_name or "").strip().rsplit(".", 1)[-1]
    candidate = re.sub(r"(?<!^)(?=[A-Z])", "_", candidate).lower()
    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate).strip("_")
    return candidate or "logger"


def _format_message(message: Any, args: tuple[Any, ...]) -> Any:
    if not args:
        return message
    if not isinstance(message, str):
        return " ".join([str(message), *[str(item) for item in args]])
    try:
        return message % args
    except Exception:
        try:
            return message.format(*args)
        except Exception:
            return " ".join([message, *[str(item) for item in args]])


def bind_plugin_logger(plugin: Any) -> None:
    """注册当前插件实例，供模块级 logger 复用 ``plugin.logger``。"""

    global _PLUGIN_INSTANCE
    _PLUGIN_INSTANCE = plugin


def _get_fallback_logger(component_name: str) -> logging.Logger:
    """插件实例尚未就绪时的兜底 logger。"""

    return logging.getLogger(f"{_FALLBACK_LOGGER_PREFIX}.{component_name}")


class _LoggerProxy:
    """兼容现有 ``logging`` 风格调用，并优先委托给 ``plugin.logger``。"""

    def __init__(self, component_name: str):
        self._component_name = component_name

    def _resolve_target(self) -> tuple[Any, bool]:
        plugin = _PLUGIN_INSTANCE
        if plugin is None:
            return _get_fallback_logger(self._component_name), False

        def patch_record(record: dict[str, Any]) -> None:
            record["name"] = f"plugin.{plugin.key}.{self._component_name}"

        return plugin.logger.patch(patch_record), True

    def _call(self, method_name: str, message: Any, *args: Any, **kwargs: Any) -> Any:
        exc_info = bool(kwargs.pop("exc_info", False))
        formatted_message = _format_message(message, args)
        target, uses_plugin_logger = self._resolve_target()

        if uses_plugin_logger:
            actual_method = "error" if method_name == "exception" else method_name
            return getattr(
                target.opt(
                    depth=2,
                    exception=exc_info or method_name == "exception",
                ),
                actual_method,
            )(formatted_message)

        actual_method = "exception" if method_name == "exception" else method_name
        if actual_method == "success":
            actual_method = "info"
        return getattr(target, actual_method)(
            formatted_message,
            exc_info=exc_info or method_name == "exception",
        )

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("debug", message, *args, **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("info", message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("warning", message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("error", message, *args, **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("critical", message, *args, **kwargs)

    def success(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call("success", message, *args, **kwargs)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["exc_info"] = True
        return self._call("exception", message, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        target, _ = self._resolve_target()
        return getattr(target, name)


def get_logger(name: str) -> Any:
    """获取兼容旧调用方式的组件 logger。"""

    component_name = _format_component_name(name)
    return _LoggerProxy(component_name)
