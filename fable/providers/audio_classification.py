"""Audio-event classification backends and the ``audio_event`` predicate provider."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Protocol, Sequence
from .data_models import AudioWindow
from .predicate_result import PredicateMatch
from .object_detection import OptionalProviderDependency

DEFAULT_AUDIO_ALIASES={
    "gunshot":("Gunshot, gunfire","Machine gun","Fusillade"),
    "alarm":("Alarm","Alarm clock","Fire alarm","Siren"),
}
class AudioBackend(Protocol):
    backend_id:str; backend_version:str
    def score(self,window:AudioWindow)->Mapping[str,float]:...
class DeterministicAudioBackend:
    backend_id="deterministic_audio"; backend_version="1"
    def __init__(self,scores:Mapping[str,float]):self.scores=dict(scores)
    def score(self,window:AudioWindow)->Mapping[str,float]:return self.scores
class YamNetBackend:
    backend_id="yamnet"; backend_version="1"
    def __init__(self,*,model:Any|None=None,model_handle:str="https://tfhub.dev/google/yamnet/1",class_names:Sequence[str]=()):self._model=model;self.model_handle=model_handle;self.class_names=tuple(class_names)
    def _load(self)->Any:
        if self._model is not None:
            self._load_class_names(self._model)
            return self._model
        try: import tensorflow_hub as hub
        except ImportError as exc: raise OptionalProviderDependency("YAMNet requires tensorflow-hub") from exc
        self._model=hub.load(self.model_handle)
        self._load_class_names(self._model)
        return self._model
    def _load_class_names(self,model:Any)->None:
        if self.class_names or not hasattr(model,"class_map_path"):
            return
        try:
            import csv
            path=model.class_map_path()
            if hasattr(path,"numpy"):path=path.numpy()
            if isinstance(path,bytes):path=path.decode("utf-8")
            with open(str(path),"r",encoding="utf-8") as handle:
                rows=list(csv.DictReader(handle))
            names=[]
            for row in rows:
                name=row.get("display_name") or row.get("name") or row.get("mid")
                if name:names.append(str(name))
            if names:self.class_names=tuple(names)
        except Exception:
            # Some injected/test YAMNet-compatible models do not expose a local
            # class-map file.  score() will fall back to class_N labels then.
            return
    def score(self,window:AudioWindow)->Mapping[str,float]:
        if window.sample_rate_hz!=16000:raise ValueError("YAMNet expects 16 kHz mono audio")
        try: import numpy as np
        except ImportError as exc: raise OptionalProviderDependency("YAMNet requires NumPy") from exc
        output=self._load()(np.asarray(window.samples,dtype=np.float32)); scores=output[0] if isinstance(output,(tuple,list)) else output
        if hasattr(scores,"numpy"):scores=scores.numpy()
        scores=np.asarray(scores,dtype=np.float32)
        if scores.ndim==2:scores=scores.mean(axis=0)
        names=self.class_names or tuple(f"class_{i}" for i in range(len(scores)))
        return {name:max(0.0,min(1.0,float(v))) for name,v in zip(names,scores)}
@dataclass(frozen=True,slots=True)
class AudioThreshold:
    semantic_class:str; minimum_score:float; minimum_consecutive_windows:int=1; refractory_seconds:float=0.0
class AudioEventClassifierProvider:
    provider_id="audio_event_classifier"; provider_version="1"
    def __init__(self,backend:AudioBackend,*,thresholds:Sequence[AudioThreshold]=(AudioThreshold("gunshot",.35),AudioThreshold("alarm",.30,2,1.0)),label_aliases:Mapping[str,Sequence[str]]=DEFAULT_AUDIO_ALIASES)->None:
        self.backend=backend; self.provider_version=str(getattr(backend,"backend_version","1")); self.thresholds={x.semantic_class:x for x in thresholds}; self.label_aliases={k:tuple(v) for k,v in label_aliases.items()}; self._consecutive=defaultdict(int); self._last={}
    def classify(self,window:AudioWindow,*,minimum_confidence:Mapping[str,float]|None=None)->tuple[PredicateMatch,...]:
        raw={str(k):max(0,min(1,float(v))) for k,v in self.backend.score(window).items()}; out=[]; overrides=dict(minimum_confidence or {})
        for semantic,threshold in sorted(self.thresholds.items()):
            score=max((raw.get(label,0.0) for label in self.label_aliases.get(semantic,(semantic,))),default=0.0); required=max(threshold.minimum_score,float(overrides.get(semantic,0.0))); key=(window.source_id,semantic)
            if score<required:self._consecutive[key]=0;continue
            self._consecutive[key]+=1
            if self._consecutive[key]<threshold.minimum_consecutive_windows:continue
            last=self._last.get(key)
            if last is not None and window.event_time<last+timedelta(seconds=threshold.refractory_seconds):continue
            out.append(PredicateMatch("audio_event",window.event_time,{"class":semantic},self.provider_id,(window.source_id,),score,self.provider_version));self._last[key]=window.event_time;self._consecutive[key]=0
        return tuple(out)
