from __future__ import annotations
from datetime import datetime,timedelta,timezone
from fable.language import load_predicates
from fable.providers import (
    AudioEventClassifierProvider,AudioWindow,BoundingBox,ConversationAVProvider,DeterministicAudioBackend,
    DiarizedSpeechSegment,DiarizedSpeechWindow,EntersBasicProvider,ExitsBasicProvider,FollowsLocalGeometryProvider,
    IoUTrackerProvider,MovingBasicProvider,NearGeometryProvider,PresentBasicProvider,
    BoardsPersonVehicleProvider,DisembarksPersonVehicleProvider,TransferCustodyProvider,
    Track,TrackFrame,load_provider_inventory,CURRENT_PUBLIC_PREDICATES,
)
T0=datetime(2026,8,29,12,0,tzinfo=timezone.utc)

def _track(object_id:str,*,t:datetime,cls:str="car",x:float=0,y:float=0,width:float=10,world=None,velocity=None,source="camera_1",confidence=.9):
    return Track(object_id,source,cls,confidence,BoundingBox(x,y,x+width,y+width),t,world,"site" if world is not None else None,velocity)
def _frame(t,*tracks,source="camera_1"):return TrackFrame(source,t,tuple(tracks))

def test_public_predicates_have_recovered_provider_path():assert set(load_predicates())==set(CURRENT_PUBLIC_PREDICATES)
def test_complete_original_provider_inventory_is_present():
    inventory=load_provider_inventory();assert len(inventory)==36
    assert {"vehicle_reid_descriptor","openclip_visual_descriptor","hosted_vlm_identity_comparator","gcc_phat_audio_localizer","speaker_diarization_provider"}<=set(inventory)

def test_initial_visibility_is_present_not_enters():
    p=PresentBasicProvider(absence_seconds=0);e=EntersBasicProvider(absence_seconds=0);car=_track("car",t=T0)
    assert [m.predicate for m in p.update(_frame(T0,car))]==["present"]
    assert e.update(_frame(T0,car))==()
def test_enters_and_exits_are_explicit_provider_ids():
    e=EntersBasicProvider(absence_seconds=0);x=ExitsBasicProvider(absence_seconds=0); e.update(_frame(T0));x.update(_frame(T0))
    t1=T0+timedelta(seconds=1);car=_track("car",t=t1)
    em=e.update(_frame(t1,car));x.update(_frame(t1,car));assert em[0].provider_id=="enters_basic"
    t2=T0+timedelta(seconds=2);assert x.update(_frame(t2))[0].provider_id=="exits_basic"
def test_moving():
    p=MovingBasicProvider(minimum_world_speed_mps=.5,minimum_elapsed_s=.1);p.update(_frame(T0,_track("car",t=T0,world=(0,0))))
    m=p.update(_frame(T0+timedelta(seconds=1),_track("car",t=T0+timedelta(seconds=1),world=(2,0))))
    assert m[0].predicate=="moving" and m[0].provider_id=="moving_basic"
def test_near_generic_dogs():
    p=NearGeometryProvider();a=_track("dog_a",t=T0,cls="dog",x=0);b=_track("dog_b",t=T0,cls="dog",x=12)
    assert p.evaluate(_frame(T0,a,b),max_normalized_gap=2.0)[0].predicate=="near"
def test_follows():
    p=FollowsLocalGeometryProvider();leader=_track("leader",t=T0,world=(10,0),velocity=(2,0));follower=_track("follower",t=T0,world=(5,0),velocity=(2,0))
    assert p.evaluate(_frame(T0,leader,follower),leader_id="leader",max_gap_m=10)[0].provider_id=="follows_local_geometry"
def test_disembarks_specific_provider():
    p=DisembarksPersonVehicleProvider(proximity_m=2,separation_m=4,minimum_stop_seconds=0);car=_track("car",t=T0,world=(0,0),velocity=(0,0));p.update(_frame(T0,car))
    t1=T0+timedelta(seconds=1);p.update(_frame(t1,_track("car",t=t1,world=(0,0),velocity=(0,0)),_track("person",t=t1,cls="person",world=(1,0))))
    t2=T0+timedelta(seconds=2);m=p.update(_frame(t2,_track("car",t=t2,world=(0,0),velocity=(0,0)),_track("person",t=t2,cls="person",world=(5,0))))
    assert m[0].provider_id=="disembarks_person_vehicle"
def test_boards_specific_provider():
    p=BoardsPersonVehicleProvider(proximity_m=2,separation_m=4,vehicle_departure_m=1,minimum_stop_seconds=0);p.update(_frame(T0,_track("person",t=T0,cls="person",world=(.5,0)),_track("car",t=T0,world=(0,0),velocity=(0,0))))
    t1=T0+timedelta(seconds=1);p.update(_frame(t1,_track("car",t=t1,world=(0,0),velocity=(0,0))))
    t2=T0+timedelta(seconds=2);m=p.update(_frame(t2,_track("car",t=t2,world=(2,0),velocity=(2,0))))
    assert m[0].provider_id=="boards_person_vehicle"
def test_transfer():
    p=TransferCustodyProvider(maximum_holder_distance_m=2,minimum_stable_seconds=0)
    def state(t,holder):
        if holder=="a":return _frame(t,_track("item",t=t,cls="backpack",world=(.5,0)),_track("a",t=t,cls="person",world=(0,0)),_track("b",t=t,cls="person",world=(10,0)))
        return _frame(t,_track("item",t=t,cls="backpack",world=(10,0)),_track("a",t=t,cls="person",world=(0,0)),_track("b",t=t,cls="person",world=(10,0)))
    p.update(state(T0,"a"));p.update(state(T0+timedelta(seconds=1),"a"));p.update(state(T0+timedelta(seconds=2),"b"));m=p.update(state(T0+timedelta(seconds=3),"b"));assert m[0].provider_id=="transfer_custody"
def test_audio_event():
    p=AudioEventClassifierProvider(DeterministicAudioBackend({"Gunshot, gunfire":.9}));m=p.classify(AudioWindow("mic",T0,(0,.1)));assert m[0].provider_id=="audio_event_classifier"
def test_diarized_speech_name_is_explicit_and_conversation_uses_it():
    speech=DiarizedSpeechWindow("mic",(DiarizedSpeechSegment("s1",T0-timedelta(seconds=1),T0,.9),DiarizedSpeechSegment("s2",T0-timedelta(seconds=.5),T0,.8)))
    p=ConversationAVProvider();a=_track("a",t=T0,cls="person",world=(0,0));b=_track("b",t=T0,cls="person",world=(1,0));m=p.evaluate(_frame(T0,a,b),speech,max_distance_m=2.5);assert m[0].provider_id=="conversation_av"
