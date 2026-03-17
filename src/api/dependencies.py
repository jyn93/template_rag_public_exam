"""FastAPI dependency injection container."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.config.settings import Settings, get_settings

__all__ = ["SettingsDep"]

SettingsDep = Annotated[Settings, Depends(get_settings)]
