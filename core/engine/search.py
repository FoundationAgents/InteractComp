#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import aiohttp
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.engine.logs import logger


class SearchEngine(ABC):
    @classmethod
    def load_config(cls, config_path: str = "config/infra_config.yaml") -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Search config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        ...

    @staticmethod
    def format_for_agent(results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No results"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title','Untitled')} — {r.get('snippet','')} (src: {r.get('source','')})")
        return "\n".join(lines)


class GoogleSerper(SearchEngine):
    def __init__(self, config: Dict[str, Any]):
        s_cfg = (config.get("search") or {})
        g_cfg = (s_cfg.get("engines") or {}).get("google", {})
        self.api_key = g_cfg.get("api_key")
        if not self.api_key:
            raise ValueError("Missing search.engines.google.api_key in config")
        r_cfg = s_cfg.get("request_settings") or {}
        self.timeout = r_cfg.get("timeout", 30)
        self.max_results = r_cfg.get("max_results_per_query", 5)
        self.endpoint = "https://google.serper.dev/search"

    async def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    data = await resp.json(content_type=None)
                    resp.raise_for_status()
        except Exception as e:
            logger.error(f"Serper error: {e}")
            return [{"title": "Search error", "snippet": str(e), "source": "serper"}]

        organic = data.get("organic") or []
        results: List[Dict[str, Any]] = []
        for item in organic[: self.max_results]:
            results.append({
                "title": item.get("title", "Untitled"),
                "snippet": item.get("snippet", ""),
                "source": item.get("link", ""),
            })
        return results


def create_search_engine(config_path: str = "config/infra_config.yaml") -> Optional[SearchEngine]:
    cfg = SearchEngine.load_config(config_path)
    engines = (cfg.get("search") or {}).get("engines") or {}
    if "google" in engines:
        return GoogleSerper(cfg)
    return None
