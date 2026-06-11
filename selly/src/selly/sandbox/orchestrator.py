"""Orchestrator pipeline F1 sandbox — coordina scraping → NLP → pricing → arbitrage → log."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

from .scrapers.vinted import VintedScraper
from .scrapers.google_trends import GoogleTrendsScraper
from .scrapers.tiktok import TikTokScraper
from .scrapers.x_search import XSearchScraper
from .nlp_extractor import NlpExtractor
from .price_discovery import PriceDiscovery
from .arbitrage_engine import compute_arbitrage
from .mock_execution import TradeLog, create_mock_trade
from .feedback_simulator import FeedbackSimulator
from .statistical_analysis import analyze


def _load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _keywords_for_nicchia(cfg: dict, nicchia: str) -> list[str]:
    return cfg.get("nicchie", {}).get(nicchia, {}).get("keywords", [])


async def run_pipeline(
    nicchia: str,
    n_trades: int,
    output: Path,
    config: dict,
) -> dict:
    """Pipeline principale F1 sandbox."""
    run_ts = datetime.now(tz=timezone.utc).isoformat()
    print(f"[F1] avvio pipeline — nicchia={nicchia}, target={n_trades} trade, ts={run_ts}")

    # 1. Perception: trend scraping — supporta 'all' per iterare su tutte le nicchie
    if nicchia == "all":
        all_nicchie = list(config.get("nicchie", {}).keys())
        keywords = []
        for n in all_nicchie:
            keywords.extend(_keywords_for_nicchia(config, n))
        print(f"  [nicchia=all] keyword totali: {len(keywords)} da {all_nicchie}")
    else:
        keywords = _keywords_for_nicchia(config, nicchia)
    if not keywords:
        raise ValueError(f"nicchia '{nicchia}' non trovata in config")

    trend_scrapers = [
        VintedScraper(rate_limit_s=config["scrapers"]["vinted"]["rate_limit_s"]),
        GoogleTrendsScraper(rate_limit_s=config["scrapers"]["google_trends"]["rate_limit_s"]),
        TikTokScraper(rate_limit_s=config["scrapers"]["tiktok"]["rate_limit_s"]),
        XSearchScraper(rate_limit_s=config["scrapers"]["x_search"]["rate_limit_s"]),
    ]

    all_signals = []
    for scraper in trend_scrapers:
        if not config["scrapers"].get(scraper.source_name.split("_")[0], {}).get("enabled", True):
            continue
        try:
            print(f"  [scraper] {scraper.source_name} ...")
            signals = await scraper.fetch_trends(keywords)
            all_signals.extend(signals)
        except Exception as e:
            print(f"  [WARN] {scraper.source_name} fallito: {e}", file=sys.stderr)

    # 2. NLP intent: estrai prodotti virali
    print(f"[F1] NLP extraction su {len(all_signals)} segnali ...")
    extractor = NlpExtractor()
    # aggrega keyword con score > 20 come priorità
    hot_kws = [s.keyword for s in all_signals if s.score >= 20.0]
    if not hot_kws:
        hot_kws = keywords  # fallback: usa tutte
    products = await extractor.extract(list(dict.fromkeys(hot_kws)))  # deduplica ordine
    print(f"  → {len(products)} prodotti estratti")

    # 3-4. Price discovery + arbitrage per ogni prodotto
    discovery = PriceDiscovery()
    log = TradeLog(output.parent / "trades.jsonl")
    opportunities = []

    for prod in products:
        if len(log) >= n_trades:
            break
        print(f"  [pricing] {prod.name} (conf={prod.confidence:.2f}) ...")
        try:
            agg = await discovery.discover(prod.name)
            opp = compute_arbitrage(
                agg,
                threshold=config["arbitrage"]["threshold_roi"],
                fees_rate=config["arbitrage"]["fees_marketplace_rate"],
                shipping_eur=config["arbitrage"]["shipping_eur"],
                vat_rate=config["arbitrage"]["vat_rate_it"],
            )
            print(f"    score={opp.score:.3f}, passes={opp.passes_threshold}")
            opportunities.append(opp)

            # 5. Mock execution
            if opp.passes_threshold and len(log) < n_trades:
                src_low = agg.price_points_low[0].source if agg.price_points_low else "bigbuy"
                src_high = agg.price_points_high[0].source if agg.price_points_high else "ebay_it"
                trade = create_mock_trade(opp, source_low=src_low, source_high=src_high)
                log.append(trade)
                print(f"    [TRADE] #{len(log)} logged: {trade.trade_id[:8]}")
                await extractor.store_in_vesta_belief(prod)
        except Exception as e:
            print(f"  [WARN] pricing fallito per '{prod.name}': {e}", file=sys.stderr)

    # Se ancora sotto il target, aggiungi trade con score < threshold (per mantenere n=30)
    if len(log) < n_trades:
        print(f"[F1] {len(log)}/{n_trades} trade con threshold — aggiungo sub-threshold per target statistico")
        for opp in sorted(opportunities, key=lambda o: o.score, reverse=True):
            if len(log) >= n_trades:
                break
            if opp.avg_price_low > 0 and opp.avg_price_high > 0:
                src_low = "bigbuy"
                src_high = "ebay_it"
                trade = create_mock_trade(opp, source_low=src_low, source_high=src_high)
                log.append(trade)
                print(f"    [TRADE sub-threshold] #{len(log)} logged")

    print(f"[F1] {len(log)} trade simulati. Avvio feedback simulation ...")

    # 6. Feedback simulation (T+14d snapshot)
    simulator = FeedbackSimulator()
    feedback_results = await simulator.evaluate_batch(log.trades[:n_trades])

    # Aggiorna actual_roi nei trade log
    for trade, fb in zip(log.trades, feedback_results):
        trade.actual_roi = fb.actual_roi
        trade.prediction_error = fb.prediction_error

    # 7. Statistical analysis
    report = analyze(feedback_results)
    print(f"[F1] calibration: accuracy={report.prediction_accuracy_rate}, MAE={report.mean_abs_error}, p={report.sign_test_p}, PASS={report.calibration_pass}")

    # Output JSON finale
    trades_serialized = []
    for trade, fb in zip(log.trades[:n_trades], feedback_results):
        d = trade.to_dict()
        d["actual_roi"] = fb.actual_roi
        d["prediction_error"] = fb.prediction_error
        trades_serialized.append(d)

    result = {
        "run_ts": run_ts,
        "nicchia": nicchia,
        "n_trades_executed": len(trades_serialized),
        "n_trades_target": n_trades,
        "prediction_accuracy_rate": report.prediction_accuracy_rate,
        "mean_abs_error": report.mean_abs_error,
        "mean_error": report.mean_error,
        "bootstrap_CI_95": list(report.bootstrap_CI_95),
        "sign_test_p": report.sign_test_p,
        "calibration_pass": report.calibration_pass,
        "trades": trades_serialized,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[F1] output scritto in {output}")
    return result


@click.command()
@click.option("--nicchia", default="salute", show_default=True, help="Nicchia target (salute|sport|beauty|all)")
@click.option("--n-trades", default=30, show_default=True, help="Numero trade simulati target")
@click.option("--output", default=None, help="Path output JSON (default: data/sandbox_results/run_<ts>.json)")
@click.option("--config", "config_path", default=None, help="Path config YAML")
def main(nicchia: str, n_trades: int, output: str | None, config_path: str | None) -> None:
    """Esegui pipeline sandbox F1 — simulazione arbitrage €0."""
    cfg_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config" / "sandbox.yaml"
    cfg = _load_config(cfg_path)

    ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(output) if output else Path(f"data/sandbox_results/run_{ts_str}.json")

    asyncio.run(run_pipeline(nicchia=nicchia, n_trades=n_trades, output=out_path, config=cfg))


if __name__ == "__main__":
    main()
