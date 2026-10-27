"""Implementations of present/enters/exits from detector-backed tracks."""
from __future__ import annotations
from datetime import datetime,timedelta
from math import hypot
from ..data_models import Track,TrackFrame
from ..predicate_result import PredicateMatch
from ._result import make_match

class _VisibilityState:
    def __init__(self,absence_seconds:float=1.5,minimum_track_age_frames:int=3,minimum_visible_seconds:float=1.5,boundary_margin:float=.25,present_minimum_confidence:float=.5)->None:
        self.absence=timedelta(seconds=absence_seconds); self.minimum_track_age_frames=minimum_track_age_frames
        self.minimum_visible=timedelta(seconds=minimum_visible_seconds); self.boundary_margin=boundary_margin; self.present_minimum_confidence=present_minimum_confidence
        self.initialized=set(); self.active={}; self.first_seen={}; self.entry_candidates={}; self.entered=set(); self.presented=set(); self.missing={}; self.last_time={}
    def _boundary(self,track:Track,frame:TrackFrame):
        # Dimensionless TrackFrames are used by non-video integrations and
        # older callers. Preserve their legacy transition semantics while
        # camera-backed frames use the stricter boundary/motion checks.
        if frame.frame_width is None or frame.frame_height is None:return ("unknown",0.0,0.0)
        x,y=track.bbox.center; distances={"left":x,"right":frame.frame_width-x,"top":y,"bottom":frame.frame_height-y}
        side=min(distances,key=distances.get); limit=(frame.frame_width if side in {"left","right"} else frame.frame_height)*self.boundary_margin
        return (side,distances[side],max(5.0,(track.bbox.x2-track.bbox.x1 if side in {"left","right"} else track.bbox.y2-track.bbox.y1)*.5)) if distances[side]<=limit else None
    def _near_boundary(self,track:Track,frame:TrackFrame)->bool:return self._boundary(track,frame) is not None
    def _looks_like_replacement(self,track:Track,frame:TrackFrame,current_ids:set[str])->bool:
        for (source_id,object_id),previous in self.active.items():
            if source_id!=frame.source_id or object_id==track.object_id or previous.class_name.lower()!=track.class_name.lower():continue
            # A still-visible track and a new track are two concurrent detector
            # observations, not two IDs for one temporarily lost object.
            if object_id in current_ids:continue
            gap=hypot(track.bbox.center[0]-previous.bbox.center[0],track.bbox.center[1]-previous.bbox.center[1])
            if gap<=max(track.bbox.width,previous.bbox.width,track.bbox.height,previous.bbox.height):return True
        return False
    def update(self,frame:TrackFrame)->tuple[tuple[Track,...],tuple[Track,...],tuple[Track,...]]:
        last=self.last_time.get(frame.source_id)
        if last is not None and frame.event_time<last: raise ValueError("visibility input must be monotonic per source")
        self.last_time[frame.source_id]=frame.event_time; current={t.object_id:t for t in frame.tracks}
        present=[]; newly=[]; exited=[]; initialized=frame.source_id in self.initialized
        if not initialized:self.initialized.add(frame.source_id)
        active_ids={oid for (src,oid) in self.active if src==frame.source_id}
        new_ids=set(current)-active_ids
        for oid,t in current.items():
            key=(frame.source_id,oid)
            if oid in new_ids:
                self.first_seen[key]=frame.event_time
                if initialized and not self._looks_like_replacement(t,frame,set(current)):
                    # Require real motion, but do not scale it from the longest
                    # side: a nearby side-on vehicle can have a very wide box
                    # and would otherwise take many seconds to qualify.
                    self.entry_candidates[key]=(
                        t.bbox.center,
                        max(5.0,min(t.bbox.width,t.bbox.height)*.25),
                    )
            self.active[key]=t; self.missing.pop(key,None)
            # ``present`` is an occurrence, not a per-frame telemetry stream.
            # Re-emitting it for every track update creates a new CE candidate
            # each time and can make the controller fall behind live sources.
            if key not in self.presented and t.confidence>=self.present_minimum_confidence:
                present.append(t); self.presented.add(key)
            candidate=self.entry_candidates.get(key)
            if candidate is not None and key not in self.entered and t.age_frames>=self.minimum_track_age_frames:
                initial_position,minimum_motion=candidate
                moved=hypot(t.bbox.center[0]-initial_position[0],t.bbox.center[1]-initial_position[1])>=minimum_motion
                # Dimensionless callers cannot express camera motion and keep
                # the historical age-gated transition behavior.
                if moved or frame.frame_width is None or frame.frame_height is None:newly.append(t); self.entered.add(key)
        for key,last_track in tuple(self.active.items()):
            src,oid=key
            if src!=frame.source_id or oid in current:continue
            since=self.missing.setdefault(key,frame.event_time)
            if frame.event_time-since<self.absence:continue
            visible=last_track.event_time-self.first_seen.get(key,last_track.event_time)
            if visible>=self.minimum_visible and self._near_boundary(last_track,frame):exited.append(last_track)
            self.active.pop(key,None); self.missing.pop(key,None); self.first_seen.pop(key,None); self.entry_candidates.pop(key,None); self.entered.discard(key)
            self.presented.discard(key)
        return tuple(present),tuple(newly),tuple(exited)

class _VisibilityPredicateProvider:
    predicate=""; provider_id=""; provider_version="1"
    def __init__(self,*,absence_seconds:float=1.5,minimum_track_age_frames:int=3,minimum_visible_seconds:float=1.5,boundary_margin:float=.25)->None:
        self._state=_VisibilityState(absence_seconds,minimum_track_age_frames,minimum_visible_seconds,boundary_margin)
    def update(self,frame:TrackFrame)->tuple[PredicateMatch,...]:
        present,entered,exited=self._state.update(frame); selected={"present":present,"enters":entered,"exits":exited}[self.predicate]
        return tuple(make_match(self.predicate,frame.event_time,{"object":t.object_id},self,(frame.source_id,),t.confidence,classes={"object":t.class_name}) for t in selected)

class PresentBasicProvider(_VisibilityPredicateProvider):
    predicate="present"; provider_id="present_basic"
class EntersBasicProvider(_VisibilityPredicateProvider):
    predicate="enters"; provider_id="enters_basic"
class ExitsBasicProvider(_VisibilityPredicateProvider):
    predicate="exits"; provider_id="exits_basic"
