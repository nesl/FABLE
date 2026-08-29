"""Multi-object tracking providers."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from .data_models import Detection, DetectionFrame, Track, TrackFrame, BoundingBox
from .object_detection import OptionalProviderDependency

@dataclass(slots=True)
class _TrackState:
    local_id:int; class_name:str; bbox:BoundingBox; confidence:float; event_time:datetime
    world_xy:tuple[float,float]|None; world_frame:str|None; age_frames:int=1; missed_frames:int=0

class IoUTrackerProvider:
    """Dependency-light local tracker used by tests and minimal deployments."""
    provider_id="iou_tracker"; provider_version="1"
    def __init__(self, *, minimum_iou:float=.25, maximum_missed_frames:int=5)->None:
        self.minimum_iou=minimum_iou; self.maximum_missed_frames=maximum_missed_frames; self._states={}; self._next_id={}; self._last_time={}
    def reset(self,source_id:str|None=None)->None:
        if source_id is None: self._states.clear(); self._next_id.clear(); self._last_time.clear(); return
        self._states.pop(source_id,None); self._next_id.pop(source_id,None); self._last_time.pop(source_id,None)
    def update(self,frame:DetectionFrame)->TrackFrame:
        last=self._last_time.get(frame.source_id)
        if last is not None and frame.event_time<last: raise ValueError("tracker inputs must be monotonic")
        self._last_time[frame.source_id]=frame.event_time; states=self._states.setdefault(frame.source_id,{})
        candidates=[]
        for tid,state in states.items():
            for di,det in enumerate(frame.detections):
                if state.class_name.lower()!=det.class_name.lower(): continue
                score=state.bbox.iou(det.bbox)
                if score>=self.minimum_iou: candidates.append((score,tid,di))
        candidates.sort(reverse=True); at=set(); ad=set(); matches={}
        for _,tid,di in candidates:
            if tid in at or di in ad: continue
            at.add(tid); ad.add(di); matches[tid]=di
        output=[]
        for tid,di in matches.items():
            det=frame.detections[di]; state=states[tid]; vel=_velocity(state,det,frame.event_time)
            state.class_name=det.class_name; state.bbox=det.bbox; state.confidence=det.confidence; state.event_time=frame.event_time; state.world_xy=det.world_xy; state.world_frame=det.world_frame; state.age_frames+=1; state.missed_frames=0
            output.append(_to_track(frame.source_id,state,vel))
        for di,det in enumerate(frame.detections):
            if di in ad: continue
            lid=self._next_id.get(frame.source_id,1); self._next_id[frame.source_id]=lid+1
            state=_TrackState(lid,det.class_name,det.bbox,det.confidence,frame.event_time,det.world_xy,det.world_frame); states[lid]=state; output.append(_to_track(frame.source_id,state,None))
        current_ids={t.object_id for t in output}
        for tid in tuple(states):
            if tid in at or _object_id(frame.source_id,tid) in current_ids: continue
            states[tid].missed_frames+=1
            if states[tid].missed_frames>self.maximum_missed_frames: states.pop(tid,None)
        return TrackFrame(frame.source_id,frame.event_time,tuple(sorted(output,key=lambda t:t.object_id)))

class MultiObjectTrackerProvider:
    """Public tracker facade matching the old ``multi_object_tracker`` provider.

    ``algorithm='iou'`` is always available. ``algorithm='bytetrack'`` uses
    Roboflow Supervision when installed, while keeping the same DetectionFrame ->
    TrackFrame contract.
    """
    provider_id="multi_object_tracker"; provider_version="1"
    def __init__(self, *, algorithm:str="iou", **kwargs:Any)->None:
        self.algorithm=algorithm.lower()
        if self.algorithm=="iou": self._tracker=IoUTrackerProvider(**kwargs)
        elif self.algorithm=="bytetrack": self._tracker=_SupervisionByteTrackProvider(**kwargs)
        else: raise ValueError("algorithm must be 'iou' or 'bytetrack'")
    def update(self,frame:DetectionFrame)->TrackFrame: return self._tracker.update(frame)

class _SupervisionByteTrackProvider:
    def __init__(self, **_:Any)->None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover
            raise OptionalProviderDependency("supervision is required for ByteTrack") from exc
        self._sv=sv; self._tracker=sv.ByteTrack()
    def update(self,frame:DetectionFrame)->TrackFrame:
        try:
            import numpy as np
        except ImportError as exc: raise OptionalProviderDependency("NumPy is required for ByteTrack") from exc
        xyxy=np.asarray([[d.bbox.x1,d.bbox.y1,d.bbox.x2,d.bbox.y2] for d in frame.detections],dtype=float)
        conf=np.asarray([d.confidence for d in frame.detections],dtype=float); class_id=np.arange(len(frame.detections),dtype=int)
        detections=self._sv.Detections(xyxy=xyxy,confidence=conf,class_id=class_id); tracked=self._tracker.update_with_detections(detections)
        out=[]
        for i,(box,tid) in enumerate(zip(tracked.xyxy,tracked.tracker_id)):
            # Supervision may reorder boxes; use nearest current detector box for class/confidence.
            bbox=BoundingBox(*map(float,box)); nearest=min(frame.detections,key=lambda d:1-d.bbox.iou(bbox))
            out.append(Track(f"{frame.source_id}:track_{int(tid)}",frame.source_id,nearest.class_name,nearest.confidence,bbox,frame.event_time))
        return TrackFrame(frame.source_id,frame.event_time,tuple(out))

def _object_id(source_id:str,local_id:int)->str: return f"{source_id}:track_{local_id}"
def _to_track(source_id:str,state:_TrackState,velocity:tuple[float,float]|None)->Track:
    return Track(_object_id(source_id,state.local_id),source_id,state.class_name,state.confidence,state.bbox,state.event_time,state.world_xy,state.world_frame,velocity,state.age_frames)
def _velocity(previous:_TrackState,detection:Detection,event_time:datetime)->tuple[float,float]|None:
    dt=(event_time-previous.event_time).total_seconds()
    if dt<=0:return None
    if previous.world_xy is not None and detection.world_xy is not None and previous.world_frame==detection.world_frame: old,new=previous.world_xy,detection.world_xy
    else: old,new=previous.bbox.center,detection.bbox.center
    return ((new[0]-old[0])/dt,(new[1]-old[1])/dt)
