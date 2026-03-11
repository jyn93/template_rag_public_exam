"""FastAPI dependency injection container."""

from __future__ import annotations

from src.core.config.settings import Settings, get_settings
from fastapi import Depends
from typing import Annotated

__all__ = ["SettingsDep"]

SettingsDep = Annotated[Settings, Depends(get_settings)]
