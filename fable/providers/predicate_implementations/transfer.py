"""Package/item transfer providers."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from ..data_models import TrackFrame,InteractionEvidence
from ..predicate_result import PredicateMatch
from ._geometry import PACKAGE_LABELS,HOLDER_LABELS,label,nearest,custody_confidence,distance_m
from ._result import make_match

class InteractionEvidenceAnalyzerProvider:
    """Score frames where an item is simultaneously close to two candidate holders."""
    provider_id="interaction_evidence_analyzer"; provider_version="1"
    def __init__(self,*,maximum_distance_m:float=2.0)->None:self.maximum_distance_m=maximum_distance_m
    def evaluate(self,frame:TrackFrame)->tuple[InteractionEvidence,...]:
        items=[t for t in frame.tracks if label(t) in PACKAGE_LABELS]; holders=[t for t in frame.tracks if label(t) in HOLDER_LABELS]; out=[]
        for item in items:
            ranked=sorted(((distance_m(item,h),h) for h in holders),key=lambda x:x[0])
            if len(ranked)<2 or ranked[1][0]>self.maximum_distance_m:continue
            score=max(0.0,1.0-ranked[1][0]/self.maximum_distance_m)
            out.append(InteractionEvidence(item.object_id,ranked[0][1].object_id,ranked[1][1].object_id,frame.event_time,score))
        return tuple(out)

@dataclass(slots=True)
class _PendingHolder: holder_id:str; since:datetime
@dataclass(slots=True)
class _Custody: holder_id:str; holder_class:str; confidence:float; established_at:datetime; last_seen_at:datetime

class TransferCustodyProvider:
    provider_id="transfer_custody"; provider_version="1"
    def __init__(self,*,maximum_holder_distance_m:float=1.5,minimum_stable_seconds:float=1.0,missing_timeout_seconds:float=5.0)->None:
        self.maximum_holder_distance_m=maximum_holder_distance_m; self.minimum_stable_seconds=minimum_stable_seconds; self.missing_timeout_seconds=missing_timeout_seconds; self._custody={}; self._pending={}; self._last_event_time=None
    def update(self,frame:TrackFrame)->tuple[PredicateMatch,...]:
        now=frame.event_time
        if self._last_event_time is not None and now<self._last_event_time:raise ValueError("transfer input must be ordered")
        items=sorted((t for t in frame.tracks if label(t) in PACKAGE_LABELS),key=lambda t:t.object_id); holders=sorted((t for t in frame.tracks if label(t) in HOLDER_LABELS),key=lambda t:t.object_id); visible={i.object_id for i in items}; out=[]
        for item in items:
            n=nearest(item,holders)
            if n is None or n[1]>self.maximum_holder_distance_m:self._pending.pop(item.object_id,None);continue
            holder,d=n; pending=self._pending.get(item.object_id)
            if pending is None or pending.holder_id!=holder.object_id:self._pending[item.object_id]=_PendingHolder(holder.object_id,now);continue
            if (now-pending.since).total_seconds()<self.minimum_stable_seconds:continue
            conf=min(item.confidence,holder.confidence,custody_confidence(d,self.maximum_holder_distance_m)); existing=self._custody.get(item.object_id)
            if existing is None:self._custody[item.object_id]=_Custody(holder.object_id,holder.class_name,conf,pending.since,now);continue
            if existing.holder_id==holder.object_id:existing.confidence=conf;existing.last_seen_at=now;existing.holder_class=holder.class_name;continue
            out.append(make_match("transfer",now,{"item":item.object_id,"giver":existing.holder_id,"receiver":holder.object_id},self,(frame.source_id,),min(existing.confidence,conf),classes={"item":item.class_name,"giver":existing.holder_class,"receiver":holder.class_name}));self._custody[item.object_id]=_Custody(holder.object_id,holder.class_name,conf,pending.since,now)
        for iid,c in tuple(self._custody.items()):
            if iid not in visible and (now-c.last_seen_at).total_seconds()>self.missing_timeout_seconds:self._custody.pop(iid,None);self._pending.pop(iid,None)
        self._last_event_time=now; return tuple(out)
