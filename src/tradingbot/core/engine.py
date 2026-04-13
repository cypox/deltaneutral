"""
Main trading engine — orchestrates all components:
  config → exchanges → data feed → strategy → executor → risk manager
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tradingbot.config.settings import Settings
from tradingbot.core.portfolio import Portfolio
from tradingbot.core.risk import RiskManager
from tradingbot.data.feed import MarketDataFeed
from tradingbot.data.storage import DataStore
from tradingbot.exchanges.base import ExchangeBase
from tradingbot.exchanges.factory import create_and_connect
from tradingbot.execution.executor import OrderExecutor
from tradingbot.execution.reconciliation import PositionReconciler
from tradingbot.strategy.delta_neutral import ActivePosition, DeltaNeutralStrategy
from tradingbot.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


class TradingEngine:
    """
    Top-level orchestrator for live trading.

    Lifecycle:
        1. initialize() — connect exchanges, load config, restore positions
        2. run() — main event loop: poll data → generate signals → execute → reconcile
        3. shutdown() — close positions if needed, disconnect
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._exchanges: dict[str, ExchangeBase] = {}
        self._strategy: DeltaNeutralStrategy | None = None
        self._executor: OrderExecutor | None = None
        self._risk_manager: RiskManager | None = None
        self._portfolio: Portfolio | None = None
        self._feed: MarketDataFeed | None = None
        self._store: DataStore | None = None
        self._reconciler: PositionReconciler | None = None
        self._running = False

    async def initialize(self) -> None:
        """Connect to exchanges, initialize all components."""
        setup_logging(self._config.app.log_level)
        log.info("engine_initializing")

        # Data dir
        Path(self._config.app.data_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        # Database
        self._store = DataStore(self._config.app.db_url)
        await self._store.initialize()

        # Connect exchanges
        for ex_id, ex_config in self._config.exchanges.items():
            if ex_config.enabled:
                try:
                    connector = await create_and_connect(ex_id, ex_config)
                    self._exchanges[ex_id] = connector
                except Exception as e:
                    log.error("exchange_connect_failed", exchange=ex_id, error=str(e))

        if not self._exchanges:
            raise RuntimeError("No exchanges connected")

        # Initialize components
        self._feed = MarketDataFeed()
        for exchange in self._exchanges.values():
            self._feed.add_exchange(exchange)

        self._strategy = DeltaNeutralStrategy(
            config=self._config.strategy,
            fee_config=self._config.fees,
            risk_config=self._config.risk,
            feed=self._feed,
            exchanges=self._exchanges,
        )

        self._executor = OrderExecutor(
            config=self._config.execution,
            exchanges=self._exchanges,
        )

        self._risk_manager = RiskManager(
            config=self._config.risk,
            exchanges=self._exchanges,
        )

        self._portfolio = Portfolio(self._exchanges)

        self._reconciler = PositionReconciler(
            exchanges=self._exchanges,
            position_tolerance_pct=self._config.risk.position_amount_tolerance_pct,
        )

        # Restore open positions from DB
        await self._restore_positions()

        log.info(
            "engine_initialized",
            exchanges=list(self._exchanges.keys()),
            symbols=self._config.strategy.symbols,
        )

    async def run(self) -> None:
        """Main trading loop."""
        self._running = True
        tick_interval = self._config.strategy.funding_check_interval
        reconcile_interval = self._config.execution.reconciliation_interval
        health_interval = self._config.risk.health_check_interval

        tick_count = 0

        log.info("engine_started", tick_interval=tick_interval)

        while self._running:
            try:
                # 1. Poll market data
                await self._feed.poll_once(self._config.strategy.symbols)

                # 2. Risk checks
                if tick_count % max(1, health_interval // tick_interval) == 0:
                    await self._run_health_checks()

                if self._risk_manager and self._risk_manager.is_halted:
                    log.warning("trading_halted_by_risk_manager")
                    await asyncio.sleep(tick_interval)
                    tick_count += 1
                    continue

                # 3. Strategy tick → signals
                if self._strategy:
                    signals = await self._strategy.on_tick()

                    # 4. Execute signals
                    for signal in signals:
                        if self._executor and not self._executor.is_circuit_broken:
                            result = await self._executor.execute_signal(signal)
                            if result and signal.is_entry:
                                await self._persist_position(result)
                                self._strategy.register_position(result)
                            elif signal.is_exit:
                                pos_key = signal.metadata.get("position_key", "")
                                parts = pos_key.split(":") if pos_key else []
                                if len(parts) == 3:
                                    self._strategy.remove_position(*parts)

                # 5. Reconciliation
                if tick_count % max(1, reconcile_interval // tick_interval) == 0:
                    await self._run_reconciliation()

                # 6. Portfolio snapshot
                if self._portfolio:
                    snap = await self._portfolio.snapshot()
                    if self._risk_manager:
                        alerts = self._risk_manager.update_equity(snap.total_equity)
                        for alert in alerts:
                            log.warning("risk_alert", alert=str(alert))

                tick_count += 1

            except KeyboardInterrupt:
                log.info("keyboard_interrupt")
                break
            except Exception as e:
                log.error("tick_error", error=str(e))
                if self._risk_manager:
                    alerts = self._risk_manager.record_error()
                    for alert in alerts:
                        log.error("risk_alert", alert=str(alert))

            await asyncio.sleep(tick_interval)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("engine_shutting_down")

        for exchange in self._exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                log.error("exchange_close_error", error=str(e))

        if self._store:
            await self._store.close()

        log.info("engine_stopped")

    # ─── Internal ────────────────────────────────────────────────────

    async def _restore_positions(self) -> None:
        """Restore open positions from database."""
        if not self._store or not self._strategy:
            return

        open_positions = await self._store.get_open_positions(self._strategy.name)
        for p in open_positions:
            pos = ActivePosition(
                db_id=p["id"],
                symbol=p["symbol"],
                spot_exchange=p["spot_exchange"],
                perp_exchange=p["perp_exchange"],
                spot_amount=p["spot_amount"],
                perp_amount=p["perp_amount"],
                spot_entry_price=p["spot_entry_price"],
                perp_entry_price=p["perp_entry_price"],
                entry_funding_rate=p["entry_funding_rate"],
                total_funding_collected=p["total_funding_collected"],
                total_fees=p["total_fees"],
            )
            self._strategy.register_position(pos)
            log.info("position_restored", symbol=pos.symbol, db_id=pos.db_id)

    async def _persist_position(self, pos: ActivePosition) -> None:
        """Save a new position to database."""
        if not self._store or not self._strategy:
            return

        db_id = await self._store.insert_position({
            "strategy": self._strategy.name,
            "symbol": pos.symbol,
            "spot_exchange": pos.spot_exchange,
            "perp_exchange": pos.perp_exchange,
            "spot_amount": pos.spot_amount,
            "perp_amount": pos.perp_amount,
            "spot_entry_price": pos.spot_entry_price,
            "perp_entry_price": pos.perp_entry_price,
            "entry_funding_rate": pos.entry_funding_rate,
            "total_funding_collected": pos.total_funding_collected,
            "total_fees": pos.total_fees,
        })
        pos.db_id = db_id

    async def _run_health_checks(self) -> None:
        if self._risk_manager:
            alerts = await self._risk_manager.check_margin_health()
            for alert in alerts:
                log.warning("health_alert", alert=str(alert))

    async def _run_reconciliation(self) -> None:
        if self._reconciler and self._strategy:
            result = await self._reconciler.reconcile(self._strategy._active_positions)
            if not result.ok:
                log.warning("reconciliation_failed", summary=result.summary())
