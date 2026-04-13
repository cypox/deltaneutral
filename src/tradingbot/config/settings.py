"""Configuration management."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return pattern.sub(replacer, value)


def _walk_and_resolve(obj: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


class ExchangeConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    secret: str = ""
    password: str = ""
    sandbox: bool = False
    rate_limit: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    name: str = "delta_neutral_funding"
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    min_funding_rate_apr: float = 0.10
    max_entry_spread_pct: float = 0.003
    max_position_usd: float = 10000.0
    min_position_usd: float = 10.0  # lowered to support small accounts ($500)
    max_total_exposure_usd: float = 50000.0
    funding_check_interval: int = 300
    exit_funding_rate_apr: float = 0.02
    max_leverage: int = 3


class FeeConfig(BaseModel):
    overrides: dict[str, float] = Field(default_factory=dict)
    slippage_bps: float = 5.0
    include_withdrawal_fees: bool = True


class RiskConfig(BaseModel):
    max_drawdown_pct: float = 0.05
    max_position_loss_pct: float = 0.03
    margin_ratio_alert: float = 0.7
    max_funding_rate_apr: float = 2.0
    min_free_margin_usd: float = 5000.0
    health_check_interval: int = 60
    max_consecutive_errors: int = 5
    position_amount_tolerance_pct: float = 0.01


class ExecutionConfig(BaseModel):
    default_order_type: str = "limit"
    limit_offset_bps: float = 2.0
    max_retries: int = 3
    limit_order_timeout: int = 30
    twap_threshold_usd: float = 5000.0
    twap_slices: int = 5
    twap_interval: int = 10
    reconciliation_interval: int = 120


class BacktestConfig(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    initial_capital: float = 100000.0
    data_source: str = "exchange"
    timeframe: str = "1h"
    funding_interval_hours: int = 8


class AppConfig(BaseModel):
    name: str = "tradingbot"
    log_level: str = "INFO"
    data_dir: str = "./data"
    db_url: str = "sqlite+aiosqlite:///./data/tradingbot.db"


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    exchanges: dict[str, ExchangeConfig] = Field(default_factory=dict)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        resolved = _walk_and_resolve(raw)
        return cls(**resolved)

    @classmethod
    def default(cls) -> Settings:
        return cls()
