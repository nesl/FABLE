"""Motion and pairwise spatial predicate implementations."""
from __future__ import annotations
from math import hypot
from ..data_models import Track, TrackFrame
from ..predicate_result import PredicateMatch
from ._geometry import distance_m, dot, euclidean, has_common_world, near
from ._result import make_match

class MovingBasicProvider:
    provider_id="moving_basic"; provider_version="1"
    def __init__(self,*,minimum_world_speed_mps:float=.75,minimum_normalized_speed_per_s:float=.5,minimum_elapsed_s:float=.2)->None:
        if min(minimum_world_speed_mps,minimum_normalized_speed_per_s,minimum_elapsed_s)<0: raise ValueError("moving thresholds cannot be negative")
        self.minimum_world_speed_mps=minimum_world_speed_mps; self.minimum_normalized_speed_per_s=minimum_normalized_speed_per_s; self.minimum_elapsed_s=minimum_elapsed_s; self._previous:dict[str,Track]={}
    def update(self,frame:TrackFrame)->tuple[PredicateMatch,...]:
        out=[]
        for track in frame.tracks:
            prev=self._previous.get(track.object_id); self._previous[track.object_id]=track
            if prev is None:continue
            elapsed=(track.event_time-prev.event_time).total_seconds()
            if elapsed<self.minimum_elapsed_s:continue
            if has_common_world(prev,track): moving=euclidean(prev.world_xy,track.world_xy)/elapsed>=self.minimum_world_speed_mps  # type: ignore[arg-type]
            else: moving=euclidean(prev.bbox.center,track.bbox.center)/max((prev.bbox.width+track.bbox.width)/2,1.0)/elapsed>=self.minimum_normalized_speed_per_s
            if moving: out.append(make_match("moving",frame.event_time,{"object":track.object_id},self,(frame.source_id,),min(prev.confidence,track.confidence),classes={"object":track.class_name}))
        return tuple(out)

class NearGeometryProvider:
    provider_id="near_geometry"; provider_version="1"
    def evaluate(self,frame:TrackFrame,*,object_a_id:str|None=None,object_b_id:str|None=None,max_distance_m:float|None=None,max_normalized_gap:float|None=None)->tuple[PredicateMatch,...]:
        if max_distance_m is not None and max_distance_m<0:raise ValueError("max_distance_m cannot be negative")
        if max_normalized_gap is not None and max_normalized_gap<0:raise ValueError("max_normalized_gap cannot be negative")
        if max_distance_m is None and max_normalized_gap is None:max_normalized_gap=2.5
        tracks=list(frame.tracks); lefts=[t for t in tracks if t.object_id==object_a_id] if object_a_id is not None else tracks; out=[]; seen=set()
        for left in lefts:
            rights=[t for t in tracks if t.object_id==object_b_id] if object_b_id is not None else tracks
            for right in rights:
                if left.object_id==right.object_id:continue
                pair=tuple(sorted((left.object_id,right.object_id)))
                if pair in seen:continue
                seen.add(pair)
                if not near(left,right,max_distance_m,max_normalized_gap):continue
                out.append(make_match("near",frame.event_time,{"object_a":left.object_id,"object_b":right.object_id},self,tuple(sorted({left.source_id,right.source_id})),min(left.confidence,right.confidence),classes={"object_a":left.class_name,"object_b":right.class_name}))
        return tuple(out)

class FollowsLocalGeometryProvider:
    provider_id="follows_local_geometry"; provider_version="1"
    def __init__(self,*,minimum_direction_cosine:float=.5)->None:
        if not -1<=minimum_direction_cosine<=1:raise ValueError("minimum_direction_cosine must be in [-1,1]")
        self.minimum_direction_cosine=minimum_direction_cosine
    def evaluate(self,frame:TrackFrame,*,leader_id:str|None=None,follower_id:str|None=None,max_gap_m:float=15.0)->tuple[PredicateMatch,...]:
        by_id=frame.by_id(); leaders=[by_id[leader_id]] if leader_id in by_id else list(frame.tracks) if leader_id is None else []; out=[]
        for leader in leaders:
            followers=[by_id[follower_id]] if follower_id is not None and follower_id in by_id else [t for t in frame.tracks if t.object_id!=leader.object_id]
            for follower in followers:
                if self._relation(leader,follower,max_gap_m): out.append(make_match("follows",frame.event_time,{"leader":leader.object_id,"follower":follower.object_id},self,tuple(sorted({leader.source_id,follower.source_id})),min(leader.confidence,follower.confidence),classes={"leader":leader.class_name,"follower":follower.class_name}))
        return tuple(out)
    def _relation(self,leader:Track,follower:Track,max_gap_m:float)->bool:
        if leader.velocity_xy_per_s is None or follower.velocity_xy_per_s is None:return False
        ls=hypot(*leader.velocity_xy_per_s); fs=hypot(*follower.velocity_xy_per_s)
        if ls<=1e-9 or fs<=1e-9:return False
        if dot(leader.velocity_xy_per_s,follower.velocity_xy_per_s)/(ls*fs)<self.minimum_direction_cosine:return False
        direction=(leader.velocity_xy_per_s[0]/ls,leader.velocity_xy_per_s[1]/ls); to_leader=(leader.position[0]-follower.position[0],leader.position[1]-follower.position[1])
        return dot(to_leader,direction)>0 and distance_m(leader,follower)<=max_gap_m

class FollowsCrossSensorProvider(FollowsLocalGeometryProvider):
    """Cross-source follows over tracks already resolved to a common coordinate frame.

    Cross-camera ReID is deliberately handled by an identity provider first; this
    evaluator only consumes the resulting tracks.
    """
    provider_id="follows_cross_sensor"
    def evaluate_pair(self,leader:Track,follower:Track,*,max_gap_m:float=15.0)->tuple[PredicateMatch,...]:
        if leader.world_frame is None or leader.world_frame!=follower.world_frame:return ()
        if not self._relation(leader,follower,max_gap_m):return ()
        t=max(leader.event_time,follower.event_time)
        return (make_match("follows",t,{"leader":leader.object_id,"follower":follower.object_id},self,tuple(sorted({leader.source_id,follower.source_id})),min(leader.confidence,follower.confidence),classes={"leader":leader.class_name,"follower":follower.class_name}),)
