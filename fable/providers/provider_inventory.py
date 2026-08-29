"""Loader for the compact provider inventory."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import yaml

@dataclass(frozen=True,slots=True)
class ProviderInfo:
    provider_id:str
    category:str
    implementation:str|tuple[str,...]
    status:str
    notes:str=""

def load_provider_inventory(path:str|Path|None=None)->dict[str,ProviderInfo]:
    p=Path(path) if path is not None else Path(__file__).with_name("provider_inventory.yaml")
    raw=yaml.safe_load(p.read_text()) or {}; providers=raw.get("providers",{})
    out={}
    for pid,spec in providers.items():
        impl=spec.get("implementation","")
        if isinstance(impl,list):impl=tuple(str(x) for x in impl)
        out[pid]=ProviderInfo(str(pid),str(spec.get("category","")),impl,str(spec.get("status","")),str(spec.get("notes","")))
    return out
