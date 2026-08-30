"""Central configuration.

Every secret (API key, project name) is read here from the `.env` file and
nowhere else. That keeps secrets out of the code and gives us one place to look
when something is misconfigured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# `.env` lives at the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Typed view of our environment. Frozen so it can't be mutated by accident."""

    aqicn_token: str
    openweather_api_key: str
    hopsworks_api_key: str | None = None
    hopsworks_project: str | None = None


def _require(name: str) -> str:
    value = os.getenv(name)

    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:
            value = None

    if not value:
        raise RuntimeError(
            f"Missing required configuration '{name}'. "
            f"Add it to your .env file locally or Streamlit Secrets when deployed."
        )

    return str(value)


def get_settings() -> Settings:
    return Settings(
        aqicn_token=_require("AQICN_TOKEN"),
        openweather_api_key=_require("OPENWEATHER_API_KEY"),
        hopsworks_api_key=os.getenv("HOPSWORKS_API_KEY") or None,
        hopsworks_project=os.getenv("HOPSWORKS_PROJECT") or None,
    )
