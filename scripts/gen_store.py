import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
logger = logging.getLogger(__name__)

class FactorStore:
    def __init__(self, config=None):
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self.config = config
        self.factors_dir = Path(config.factors_dir)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._qlib_inited = False

    def load(self, name, expr, start_date=None, end_date=None):
        r = self._load_evaluated(name, start_date, end_date)
        if r is not None: return r
        r = self._load_cache(name, start_date, end_date)
        if r is not None: return r
        return self._compute(name, expr, start_date, end_date)

    def get_evaluated(self, name, start_date=None, end_date=None):
        return self._load_evaluated(name, start_date, end_date)

    def get_evaluated_meta(self, name):
        for tier in ["core","satellite","archive"]:
            for mf in (self.factors_dir / tier).rglob(f"{name}.meta.json"):
                import json
                with open(mf) as f: return json.load(f)
        return None

    def list_evaluated(self):
        result = []
        for tier in ["core","satellite","archive"]:
            td = self.factors_dir / tier
            if not td.exists(): continue
            for mf in sorted(td.rglob("*.meta.json")):
                import json; meta=json.load(open(mf))
                result.append({"name":meta.get("factor_name",""),"tier":tier})
        return result