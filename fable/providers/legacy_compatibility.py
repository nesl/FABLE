"""Compatibility implementations for old FABLE providers no longer in CE v1.

These providers are retained so the rebuild contains an explicit home for the
old catalog rather than silently dropping algorithms. They are *not* part of the
current public CE predicate vocabulary (which intentionally removed zones,
passes, dwell, historical-before predicates, etc.).
"""
from __future__ import annotations
from datetime import datetime,timedelta
from .data_models import Track,TrackFrame
from .predicate_result import PredicateMatch

class ZoneMembershipEvaluatorProvider:
    provider_id="zone_membership_evaluator";provider_version="1"
    def evaluate(self,frame:TrackFrame,polygon:list[tuple[float,float]])->tuple[PredicateMatch,...]:
        return tuple(PredicateMatch("inside",frame.event_time,{"object":t.object_id},self.provider_id,(frame.source_id,),t.confidence,self.provider_version) for t in frame.tracks if _inside(t.position,polygon))

class ZoneTransitionEvaluatorProvider:
    provider_id="zone_transition_evaluator";provider_version="1"
    def __init__(self)->None:self._inside:set[str]=set()
    def evaluate(self,frame:TrackFrame,polygon:list[tuple[float,float]])->tuple[PredicateMatch,...]:
        now={t.object_id:t for t in frame.tracks if _inside(t.position,polygon)};out=[]
        for oid in set(now)-self._inside:out.append(PredicateMatch("enters_region",frame.event_time,{"object":oid},self.provider_id,(frame.source_id,),now[oid].confidence,self.provider_version))
        for oid in self._inside-set(now):out.append(PredicateMatch("exits_region",frame.event_time,{"object":oid},self.provider_id,(frame.source_id,),1.0,self.provider_version))
        self._inside=set(now);return tuple(out)

class PassReferenceEvaluatorProvider:
    provider_id="pass_reference_evaluator";provider_version="1"
    def __init__(self)->None:self._side={}
    def evaluate(self,frame:TrackFrame,line_a:tuple[float,float],line_b:tuple[float,float])->tuple[PredicateMatch,...]:
        out=[]
        for t in frame.tracks:
            side=_side(t.position,line_a,line_b); prev=self._side.get(t.object_id);self._side[t.object_id]=side
            if prev is not None and prev*side<0:out.append(PredicateMatch("passes",frame.event_time,{"object":t.object_id},self.provider_id,(frame.source_id,),t.confidence,self.provider_version))
        return tuple(out)

class RelativeOrderEvaluatorProvider:
    provider_id="relative_order_evaluator";provider_version="1"
    def behind(self,leader:Track,follower:Track,direction:tuple[float,float])->bool:
        dx=leader.position[0]-follower.position[0];dy=leader.position[1]-follower.position[1];return dx*direction[0]+dy*direction[1]>0

class RouteMapMatcherProvider:
    provider_id="route_map_matcher";provider_version="1"
    def match(self,track:Track,routes:dict[str,list[tuple[float,float]]])->str|None:
        if not routes:return None
        return min(routes,key=lambda rid:min((_dist(track.position,p) for p in routes[rid]),default=float("inf")))

class DwellEvaluatorProvider:
    provider_id="dwell_evaluator";provider_version="1"
    def __init__(self,*,duration_seconds:float=5.0)->None:self.duration=timedelta(seconds=duration_seconds);self._since={}
    def evaluate(self,frame:TrackFrame,polygon:list[tuple[float,float]])->tuple[PredicateMatch,...]:
        out=[]; visible=set()
        for t in frame.tracks:
            if not _inside(t.position,polygon):continue
            visible.add(t.object_id); since=self._since.setdefault(t.object_id,frame.event_time)
            if frame.event_time-since>=self.duration:out.append(PredicateMatch("dwells",frame.event_time,{"object":t.object_id},self.provider_id,(frame.source_id,),t.confidence,self.provider_version))
        for oid in set(self._since)-visible:self._since.pop(oid,None)
        return tuple(out)

class TrackSummaryRouteEvaluatorProvider:
    provider_id="track_summary_route_evaluator";provider_version="1"
    def evaluate(self,positions:list[tuple[float,float]],route:list[tuple[float,float]],*,maximum_distance:float=10.0)->bool:
        return bool(positions) and all(min((_dist(p,r) for r in route),default=float("inf"))<=maximum_distance for p in positions)

class HistoricalVehicleIntervalMatcherProvider:
    provider_id="historical_vehicle_interval_matcher";provider_version="1"
    def match(self,frames:list[TrackFrame],*,start:datetime,end:datetime)->tuple[PredicateMatch,...]:
        seen={}
        for frame in frames:
            if not start<=frame.event_time<=end:continue
            for t in frame.tracks:
                if t.class_name.lower() in {"car","truck","bus","motorcycle","vehicle"}:seen[t.object_id]=(frame,t)
        return tuple(PredicateMatch("vehicle_present_before",frame.event_time,{"object":oid},self.provider_id,(frame.source_id,),track.confidence,self.provider_version) for oid,(frame,track) in sorted(seen.items()))

def _inside(point:tuple[float,float],polygon:list[tuple[float,float]])->bool:
    x,y=point;inside=False;j=len(polygon)-1
    for i,(xi,yi) in enumerate(polygon):
        xj,yj=polygon[j]
        if ((yi>y)!=(yj>y)) and x<(xj-xi)*(y-yi)/(yj-yi+1e-12)+xi:inside=not inside
        j=i
    return inside
def _side(p,a,b):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
def _dist(a,b):return ((a[0]-b[0])**2+(a[1]-b[1])**2)**.5
