"""Person/vehicle transition predicate implementations."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from ..data_models import Track,TrackFrame
from ..predicate_result import PredicateMatch
from ._geometry import PERSON_LABELS,VEHICLE_LABELS,label,nearest,distance_m,distance_positions_m,speed_mps
from ._result import make_match

@dataclass(slots=True)
class _DisembarkCandidate:
    person_id:str; vehicle_id:str; started_at:datetime; initial_distance_m:float
@dataclass(slots=True)
class _BoardCandidate:
    person_id:str; vehicle_id:str; person_last_seen_at:datetime; vehicle_position:tuple[float,float]

class _PersonVehicleRelationProvider:
    target_predicate=""; provider_id=""; provider_version="1"
    def __init__(self,*,proximity_m:float=3.0,separation_m:float=5.0,vehicle_departure_m:float=2.0,stopped_speed_mps:float=.75,minimum_stop_seconds:float=.5,transition_window_seconds:float=8.0)->None:
        if min(proximity_m,separation_m,vehicle_departure_m,stopped_speed_mps,minimum_stop_seconds,transition_window_seconds)<0:raise ValueError("thresholds cannot be negative")
        if separation_m<=proximity_m:raise ValueError("separation_m must exceed proximity_m")
        self.proximity_m=proximity_m; self.separation_m=separation_m; self.vehicle_departure_m=vehicle_departure_m; self.stopped_speed_mps=stopped_speed_mps; self.minimum_stop_seconds=minimum_stop_seconds; self.transition_window_seconds=transition_window_seconds
        self._previous={}; self._vehicle_stopped_since={}; self._disembark={}; self._boarding={}; self._emitted=set(); self._last_event_time=None
    def update(self,frame:TrackFrame)->tuple[PredicateMatch,...]:
        now=frame.event_time
        if self._last_event_time is not None and now<self._last_event_time:raise ValueError("person/vehicle input must be ordered")
        current={t.object_id:t for t in frame.tracks}; persons={k:v for k,v in current.items() if label(v) in PERSON_LABELS}; vehicles={k:v for k,v in current.items() if label(v) in VEHICLE_LABELS}; prev_persons={k:v for k,v in self._previous.items() if label(v) in PERSON_LABELS}; prev_vehicles={k:v for k,v in self._previous.items() if label(v) in VEHICLE_LABELS}
        self._update_stops(vehicles,now); outputs=[]
        if self.target_predicate=="disembarks":
            for pid in sorted(set(persons)-set(prev_persons)):
                n=nearest(persons[pid],vehicles.values())
                if n is None:continue
                vehicle,d=n; stopped=self._vehicle_stopped_since.get(vehicle.object_id)
                if d<=self.proximity_m and stopped is not None and (now-stopped).total_seconds()>=self.minimum_stop_seconds:self._disembark[pid]=_DisembarkCandidate(pid,vehicle.object_id,now,d)
            for pid,c in tuple(self._disembark.items()):
                if (now-c.started_at).total_seconds()>self.transition_window_seconds:self._disembark.pop(pid,None);continue
                person=persons.get(pid); vehicle=vehicles.get(c.vehicle_id)
                if person is None or vehicle is None or distance_m(person,vehicle)<self.separation_m:continue
                key=(pid,c.vehicle_id)
                if key not in self._emitted:outputs.append(make_match("disembarks",now,{"person":pid,"vehicle":c.vehicle_id},self,(frame.source_id,),min(person.confidence,vehicle.confidence),classes={"person":"person","vehicle":"vehicle"}));self._emitted.add(key)
                self._disembark.pop(pid,None)
        else:
            for pid in sorted(set(prev_persons)-set(persons)):
                n=nearest(prev_persons[pid],prev_vehicles.values())
                if n is None:continue
                vehicle,d=n
                if d<=self.proximity_m:self._boarding[pid]=_BoardCandidate(pid,vehicle.object_id,prev_persons[pid].event_time,vehicle.position)
            for pid,c in tuple(self._boarding.items()):
                if (now-c.person_last_seen_at).total_seconds()>self.transition_window_seconds:self._boarding.pop(pid,None);continue
                vehicle=vehicles.get(c.vehicle_id)
                if vehicle is None:continue
                moved=distance_positions_m(c.vehicle_position,vehicle.position,vehicle); speed=speed_mps(vehicle)
                if moved<self.vehicle_departure_m and (speed is None or speed<=self.stopped_speed_mps):continue
                key=(pid,c.vehicle_id)
                if key not in self._emitted:outputs.append(make_match("boards",now,{"person":pid,"vehicle":c.vehicle_id},self,(frame.source_id,),vehicle.confidence,classes={"person":"person","vehicle":"vehicle"}));self._emitted.add(key)
                self._boarding.pop(pid,None)
        self._previous=current; self._last_event_time=now; return tuple(outputs)
    def _update_stops(self,vehicles:dict[str,Track],now:datetime)->None:
        for vid,v in vehicles.items():
            speed=speed_mps(v); stopped=speed is None or speed<=self.stopped_speed_mps
            if stopped:self._vehicle_stopped_since.setdefault(vid,now)
            else:self._vehicle_stopped_since.pop(vid,None)
        for missing in set(self._vehicle_stopped_since)-set(vehicles):self._vehicle_stopped_since.pop(missing,None)

class BoardsPersonVehicleProvider(_PersonVehicleRelationProvider):
    target_predicate="boards"; provider_id="boards_person_vehicle"
class DisembarksPersonVehicleProvider(_PersonVehicleRelationProvider):
    target_predicate="disembarks"; provider_id="disembarks_person_vehicle"
