"""Voice activity, speaker embeddings/diarization, and optional speech content."""
from __future__ import annotations
from datetime import timedelta
from math import sqrt
from typing import Callable, Iterable, Sequence
from .data_models import AudioWindow,SpeechSegment,SpeakerEmbedding,DiarizedSpeechSegment,DiarizedSpeechWindow,TranscriptSegment

class VoiceActivityDetectorProvider:
    provider_id="voice_activity_detector"; provider_version="1"
    def __init__(self,*,energy_threshold:float=.01)->None:self.energy_threshold=energy_threshold
    def detect(self,window:AudioWindow)->tuple[SpeechSegment,...]:
        if not window.samples:return ()
        rms=(sum(x*x for x in window.samples)/len(window.samples))**.5
        if rms<self.energy_threshold:return ()
        duration=len(window.samples)/window.sample_rate_hz
        return (SpeechSegment(window.event_time,window.event_time+timedelta(seconds=duration),min(1.0,rms/max(self.energy_threshold,1e-9))),)

class SpeakerEmbeddingProvider:
    provider_id="speaker_embedding_provider"; provider_version="1"
    def __init__(self,embedder:Callable[[AudioWindow,SpeechSegment],Sequence[float]]|None=None,*,model_id:str="simple_spectral"):
        self.embedder=embedder or self._simple_embedding; self.model_id=model_id
    def embed(self,window:AudioWindow,segments:Iterable[SpeechSegment])->tuple[SpeakerEmbedding,...]:
        return tuple(SpeakerEmbedding(seg,tuple(float(v) for v in self.embedder(window,seg)),self.model_id) for seg in segments)
    def _simple_embedding(self,window:AudioWindow,segment:SpeechSegment)->Sequence[float]:
        samples=window.samples
        if not samples:return (0.0,0.0,0.0)
        mean=sum(samples)/len(samples); rms=(sum(x*x for x in samples)/len(samples))**.5; zc=sum(1 for a,b in zip(samples,samples[1:]) if (a>=0)!=(b>=0))/max(len(samples)-1,1)
        return (mean,rms,zc)

class SpeakerDiarizationProvider:
    provider_id="speaker_diarization_provider"; provider_version="1"
    def __init__(self,*,cosine_threshold:float=.75)->None:self.cosine_threshold=cosine_threshold
    def diarize(self,source_id:str,embeddings:Sequence[SpeakerEmbedding])->DiarizedSpeechWindow:
        centroids=[]; counts=[]; out=[]
        for emb in embeddings:
            if not centroids: idx=0;centroids.append(list(emb.vector));counts.append(1)
            else:
                scores=[_cosine(emb.vector,c) for c in centroids]; best=max(range(len(scores)),key=scores.__getitem__)
                if scores[best]>=self.cosine_threshold:
                    idx=best; n=counts[idx]; centroids[idx]=[(a*n+b)/(n+1) for a,b in zip(centroids[idx],emb.vector)];counts[idx]+=1
                else: idx=len(centroids);centroids.append(list(emb.vector));counts.append(1)
            out.append(DiarizedSpeechSegment(f"speaker_{idx+1}",emb.segment.start_time,emb.segment.end_time,1.0))
        return DiarizedSpeechWindow(source_id,tuple(out))

class KeywordOrASRProvider:
    provider_id="keyword_or_asr_provider"; provider_version="1"
    def __init__(self,transcriber:Callable[[AudioWindow,DiarizedSpeechSegment],str]|None=None):self.transcriber=transcriber
    def transcribe(self,window:AudioWindow,diarized:DiarizedSpeechWindow)->tuple[TranscriptSegment,...]:
        out=[]
        for seg in diarized.segments:
            text=seg.transcript or (self.transcriber(window,seg) if self.transcriber is not None else "")
            if text:out.append(TranscriptSegment(seg.speaker_id,seg.start_time,seg.end_time,text))
        return tuple(out)

def _cosine(a:Sequence[float],b:Sequence[float])->float:
    if not a or len(a)!=len(b):return 0.0
    dot=sum(x*y for x,y in zip(a,b)); na=sqrt(sum(x*x for x in a)); nb=sqrt(sum(y*y for y in b)); return 0.0 if na==0 or nb==0 else dot/(na*nb)
