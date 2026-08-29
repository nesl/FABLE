"""Conversation predicate implementation."""
from __future__ import annotations
from typing import Sequence
from ..data_models import TrackFrame,DiarizedSpeechWindow,TranscriptSegment
from ..predicate_result import PredicateMatch
from ._geometry import PERSON_LABELS,label,distance_m
from ._result import make_match

class ConversationAVProvider:
    provider_id="conversation_av"; provider_version="1"
    def __init__(self,*,minimum_speakers:int=2)->None:self.minimum_speakers=minimum_speakers
    def evaluate(self,tracks:TrackFrame,speech:DiarizedSpeechWindow,*,participant_a_id:str|None=None,participant_b_id:str|None=None,max_distance_m:float=2.5,required_terms:Sequence[str]=(),transcripts:Sequence[TranscriptSegment]=())->tuple[PredicateMatch,...]:
        if speech.speaker_count<self.minimum_speakers:return ()
        terms=tuple(t.strip().lower() for t in required_terms if t.strip()); text=" ".join([s.transcript or "" for s in speech.segments]+[t.text for t in transcripts]).lower()
        if terms and not all(term in text for term in terms):return ()
        persons=[t for t in tracks.tracks if label(t) in PERSON_LABELS]; out=[];seen=set(); lefts=[p for p in persons if p.object_id==participant_a_id] if participant_a_id else persons
        for left in lefts:
            rights=[p for p in persons if p.object_id==participant_b_id] if participant_b_id else persons
            for right in rights:
                if left.object_id==right.object_id:continue
                pair=tuple(sorted((left.object_id,right.object_id)))
                if pair in seen:continue
                seen.add(pair)
                if distance_m(left,right)>max_distance_m:continue
                conf=min(left.confidence,right.confidence,min((s.confidence for s in speech.segments),default=1.0)); out.append(make_match("conversation",tracks.event_time,{"participant_a":left.object_id,"participant_b":right.object_id},self,tuple(sorted({tracks.source_id,speech.source_id})),conf,classes={"participant_a":"person","participant_b":"person"}))
        return tuple(out)
