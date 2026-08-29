"""Shared geometric helpers for predicate implementations."""
from __future__ import annotations
from math import hypot
from typing import Iterable
from ..data_models import Track

PERSON_LABELS=frozenset({"person"})
VEHICLE_LABELS=frozenset({"car","truck","bus","motorcycle","vehicle"})
PACKAGE_LABELS=frozenset({"backpack","handbag","suitcase","package","parcel","box","bag"})
HOLDER_LABELS=PERSON_LABELS|VEHICLE_LABELS
APPROXIMATE_WIDTH_M={"person":.55,"bicycle":1.8,"car":4.0,"truck":7.0,"bus":10.0,"motorcycle":2.0,"vehicle":4.0,"dog":.8,"backpack":.45,"handbag":.35,"suitcase":.55,"package":.5,"parcel":.5,"box":.5,"bag":.45}

def label(track:Track)->str:return track.class_name.lower()
def has_common_world(a:Track,b:Track)->bool:return a.world_xy is not None and b.world_xy is not None and a.world_frame is not None and a.world_frame==b.world_frame
def euclidean(a:tuple[float,float],b:tuple[float,float])->float:return hypot(a[0]-b[0],a[1]-b[1])
def mean_bbox_width(a:Track,b:Track)->float:return max((a.bbox.width+b.bbox.width)/2,1.0)
def representative_width_m(t:Track)->float:return APPROXIMATE_WIDTH_M.get(label(t),1.0)
def distance_m(a:Track,b:Track)->float:
    if has_common_world(a,b):return euclidean(a.world_xy,b.world_xy)  # type: ignore[arg-type]
    return euclidean(a.bbox.center,b.bbox.center)/mean_bbox_width(a,b)*(representative_width_m(a)+representative_width_m(b))/2
def normalized_gap(a:Track,b:Track)->float:return euclidean(a.bbox.center,b.bbox.center)/mean_bbox_width(a,b)
def near(a:Track,b:Track,max_distance_m:float|None,max_normalized_gap:float|None)->bool:
    if has_common_world(a,b):return distance_m(a,b) <= (5.0 if max_distance_m is None else max_distance_m)
    if max_normalized_gap is not None:return normalized_gap(a,b)<=max_normalized_gap
    if max_distance_m is not None:return distance_m(a,b)<=max_distance_m
    return False
def nearest(subject:Track,candidates:Iterable[Track])->tuple[Track,float]|None:
    ranked=sorted(((distance_m(subject,c),c) for c in candidates if c.object_id!=subject.object_id),key=lambda x:(x[0],x[1].object_id))
    return None if not ranked else (ranked[0][1],ranked[0][0])
def speed_mps(t:Track)->float|None:
    if t.velocity_xy_per_s is None:return None
    speed=hypot(*t.velocity_xy_per_s)
    return speed if t.world_xy is not None else speed/max(t.bbox.width,1.0)*representative_width_m(t)
def distance_positions_m(old:tuple[float,float],new:tuple[float,float],ref:Track)->float:
    raw=euclidean(old,new); return raw if ref.world_xy is not None else raw/max(ref.bbox.width,1.0)*representative_width_m(ref)
def dot(a:tuple[float,float],b:tuple[float,float])->float:return a[0]*b[0]+a[1]*b[1]
def custody_confidence(distance:float,maximum:float)->float:return max(0.0,min(1.0,.55+.45*(1-distance/maximum)))
