"""Runtime dependency probing and dynamic install helpers."""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any

from nekro_agent.api.plugin import dynamic_import_pkg

from amemorix.common.logging import get_logger

logger = get_logger("A_Memorix.RuntimeDeps")


@dataclass(frozen=True)
class RuntimeDependencySpec:
    name: str
    package_spec: str
    import_name: str
    required: bool
    provider: str
    dynamic_supported: bool = False


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    name: str
    package_spec: str
    import_name: str
    available: bool
    required: bool
    provider: str
    dynamic_supported: bool
    installed_by_plugin: bool = False
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "package_spec": self.package_spec,
            "import_name": self.import_name,
            "available": self.available,
            "required": self.required,
            "provider": self.provider,
            "dynamic_supported": self.dynamic_supported,
            "installed_by_plugin": self.installed_by_plugin,
            "detail": self.detail,
        }


_HOST_OPENAI_SPEC = RuntimeDependencySpec(
    name="OpenAI SDK",
    package_spec="openai",
    import_name="openai",
    required=True,
    provider="host_builtin",
)
_HOST_PSYCOPG2_SPEC = RuntimeDependencySpec(
    name="psycopg2-binary",
    package_spec="psycopg2-binary",
    import_name="psycopg2",
    required=True,
    provider="host_builtin",
)
_HOST_QDRANT_SPEC = RuntimeDependencySpec(
    name="qdrant-client",
    package_spec="qdrant-client",
    import_name="qdrant_client",
    required=True,
    provider="host_builtin",
)
_SCIPY_SPEC = RuntimeDependencySpec(
    name="SciPy",
    package_spec="scipy",
    import_name="scipy",
    required=True,
    provider="plugin_dynamic",
    dynamic_supported=True,
)
_JIEBA_SPEC = RuntimeDependencySpec(
    name="jieba",
    package_spec="jieba",
    import_name="jieba",
    required=False,
    provider="plugin_dynamic",
    dynamic_supported=True,
)
_SENTENCE_TRANSFORMERS_SPEC = RuntimeDependencySpec(
    name="sentence-transformers",
    package_spec="sentence-transformers",
    import_name="sentence_transformers",
    required=False,
    provider="plugin_dynamic",
    dynamic_supported=True,
)

_dependency_lock = threading.Lock()
_status_cache: dict[tuple[str, bool], RuntimeDependencyStatus] = {}


def _import_available(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
    except ImportError:
        return False
    return True


def _build_available_status(
    spec: RuntimeDependencySpec,
    *,
    installed_by_plugin: bool = False,
    detail: str = "",
) -> RuntimeDependencyStatus:
    return RuntimeDependencyStatus(
        name=spec.name,
        package_spec=spec.package_spec,
        import_name=spec.import_name,
        available=True,
        required=spec.required,
        provider=spec.provider,
        dynamic_supported=spec.dynamic_supported,
        installed_by_plugin=installed_by_plugin,
        detail=detail,
    )


def _build_unavailable_status(spec: RuntimeDependencySpec, detail: str) -> RuntimeDependencyStatus:
    return RuntimeDependencyStatus(
        name=spec.name,
        package_spec=spec.package_spec,
        import_name=spec.import_name,
        available=False,
        required=spec.required,
        provider=spec.provider,
        dynamic_supported=spec.dynamic_supported,
        installed_by_plugin=False,
        detail=detail,
    )


def _cache_status(status: RuntimeDependencyStatus, *, install_if_missing: bool) -> RuntimeDependencyStatus:
    cache_key = (status.import_name, install_if_missing)
    _status_cache[cache_key] = status
    if status.available:
        _status_cache[(status.import_name, False)] = status
        _status_cache[(status.import_name, True)] = status
    return status


def _check_dependency(
    spec: RuntimeDependencySpec,
    *,
    install_if_missing: bool,
) -> RuntimeDependencyStatus:
    cache_key = (spec.import_name, install_if_missing)
    with _dependency_lock:
        cached = _status_cache.get(cache_key)
        if cached is not None:
            return cached

        if _import_available(spec.import_name):
            detail = f"{spec.name} 已就绪。"
            return _cache_status(
                _build_available_status(spec, installed_by_plugin=False, detail=detail),
                install_if_missing=install_if_missing,
            )

        if install_if_missing and spec.dynamic_supported:
            try:
                dynamic_import_pkg(spec.package_spec, spec.import_name)
            except Exception as exc:
                detail = (
                    f"{spec.name} 不可用，已尝试通过 dynamic_import_pkg 自动安装，但失败：{exc}"
                )
                logger.warning(detail)
                return _cache_status(
                    _build_unavailable_status(spec, detail),
                    install_if_missing=install_if_missing,
                )

            if _import_available(spec.import_name):
                detail = f"{spec.name} 缺失，已通过 dynamic_import_pkg 自动安装。"
                logger.info(detail)
                return _cache_status(
                    _build_available_status(spec, installed_by_plugin=True, detail=detail),
                    install_if_missing=install_if_missing,
                )

            detail = f"{spec.name} 已尝试动态安装，但仍无法导入。"
            logger.warning(detail)
            return _cache_status(
                _build_unavailable_status(spec, detail),
                install_if_missing=install_if_missing,
            )

        if spec.dynamic_supported:
            detail = f"{spec.name} 未就绪，可在首次需要时通过 dynamic_import_pkg 自动安装。"
        elif spec.required:
            detail = f"{spec.name} 未就绪，依赖宿主环境提供。"
        else:
            detail = f"{spec.name} 未就绪。"

        return _cache_status(
            _build_unavailable_status(spec, detail),
            install_if_missing=install_if_missing,
        )


def _load_module(spec: RuntimeDependencySpec, *, install_if_missing: bool) -> Any | None:
    status = _check_dependency(spec, install_if_missing=install_if_missing)
    if not status.available:
        return None
    try:
        return importlib.import_module(spec.import_name)
    except ImportError as exc:
        detail = f"{spec.name} 已安装但导入失败：{exc}"
        logger.warning(detail)
        with _dependency_lock:
            _status_cache[(spec.import_name, False)] = _build_unavailable_status(spec, detail)
            _status_cache[(spec.import_name, True)] = _build_unavailable_status(spec, detail)
        return None


def probe_openai() -> RuntimeDependencyStatus:
    return _check_dependency(_HOST_OPENAI_SPEC, install_if_missing=False)


def probe_psycopg2() -> RuntimeDependencyStatus:
    return _check_dependency(_HOST_PSYCOPG2_SPEC, install_if_missing=False)


def probe_qdrant_client() -> RuntimeDependencyStatus:
    return _check_dependency(_HOST_QDRANT_SPEC, install_if_missing=False)


def ensure_scipy() -> RuntimeDependencyStatus:
    return _check_dependency(_SCIPY_SPEC, install_if_missing=True)


def probe_jieba() -> RuntimeDependencyStatus:
    return _check_dependency(_JIEBA_SPEC, install_if_missing=False)


def ensure_jieba() -> RuntimeDependencyStatus:
    return _check_dependency(_JIEBA_SPEC, install_if_missing=True)


def load_jieba(*, install_if_missing: bool) -> Any | None:
    return _load_module(_JIEBA_SPEC, install_if_missing=install_if_missing)


def probe_sentence_transformers() -> RuntimeDependencyStatus:
    return _check_dependency(_SENTENCE_TRANSFORMERS_SPEC, install_if_missing=False)


def ensure_sentence_transformers() -> RuntimeDependencyStatus:
    return _check_dependency(_SENTENCE_TRANSFORMERS_SPEC, install_if_missing=True)


def load_sentence_transformers(*, install_if_missing: bool) -> Any | None:
    return _load_module(_SENTENCE_TRANSFORMERS_SPEC, install_if_missing=install_if_missing)


def get_runtime_dependency_report() -> dict[str, Any]:
    items = [
        probe_openai().to_payload(),
        probe_psycopg2().to_payload(),
        probe_qdrant_client().to_payload(),
        ensure_scipy().to_payload(),
        probe_jieba().to_payload(),
        probe_sentence_transformers().to_payload(),
    ]
    missing_required = [item["name"] for item in items if item["required"] and not item["available"]]
    detail = "；".join(item["detail"] for item in items if item["required"] and not item["available"])
    return {
        "ready": not missing_required,
        "items": items,
        "missing": missing_required,
        "detail": detail,
    }
