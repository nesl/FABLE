"""Implementations of present/enters/exits from detector-backed tracks."""
from __future__ import annotations
from datetime import datetime,timedelta
from ..data_models import Track,TrackFrame
from ..predicate_result import PredicateMatch
from ._result import make_match

class _VisibilityState:
    def __init__(self,absence_seconds:float=.5)->None:
        self.absence=timedelta(seconds=absence_seconds); self.initialized=set(); self.active={}; self.missing={}; self.last_time={}
    def update(self,frame:TrackFrame)->tuple[tuple[Track,...],tuple[Track,...],tuple[Track,...]]:
        last=self.last_time.get(frame.source_id)
        if last is not None and frame.event_time<last: raise ValueError("visibility input must be monotonic per source")
        self.last_time[frame.source_id]=frame.event_time; current={t.object_id:t for t in frame.tracks}
        newly=[]; exited=[]; initialized=frame.source_id in self.initialized
        if not initialized:self.initialized.add(frame.source_id)
        active_ids={oid for (src,oid) in self.active if src==frame.source_id}
        if initialized:newly=[current[oid] for oid in sorted(set(current)-active_ids)]
        for oid,t in current.items(): self.active[(frame.source_id,oid)]=t; self.missing.pop((frame.source_id,oid),None)
        for key,last_track in tuple(self.active.items()):
            src,oid=key
            if src!=frame.source_id or oid in current:continue
            since=self.missing.setdefault(key,frame.event_time)
            if frame.event_time-since<self.absence:continue
            exited.append(last_track); self.active.pop(key,None); self.missing.pop(key,None)
        return tuple(frame.tracks),tuple(newly),tuple(exited)

class _VisibilityPredicateProvider:
    predicate=""; provider_id=""; provider_version="1"
    def __init__(self,*,absence_seconds:float=.5)->None:self._state=_VisibilityState(absence_seconds)
    def update(self,frame:TrackFrame)->tuple[PredicateMatch,...]:
        present,entered,exited=self._state.update(frame); selected={"present":present,"enters":entered,"exits":exited}[self.predicate]
        return tuple(make_match(self.predicate,frame.event_time,{"object":t.object_id},self,(frame.source_id,),t.confidence,classes={"object":t.class_name}) for t in selected)

class PresentBasicProvider(_VisibilityPredicateProvider):
    predicate="present"; provider_id="present_basic"
class EntersBasicProvider(_VisibilityPredicateProvider):
    predicate="enters"; provider_id="enters_basic"
class ExitsBasicProvider(_VisibilityPredicateProvider):
    predicate="exits"; provider_id="exits_basic"
