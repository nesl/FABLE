"""Audio localization and audio/visual association providers."""
from __future__ import annotations
from math import degrees, asin
from typing import Sequence
from .data_models import MultichannelAudioWindow,AudioLocalization,VisualBearing
from .object_detection import OptionalProviderDependency

class GccPhatAudioLocalizerProvider:
    provider_id="gcc_phat_audio_localizer"; provider_version="1"
    def __init__(self,*,microphone_spacing_m:float=.08,speed_of_sound_mps:float=343.0)->None:
        self.microphone_spacing_m=microphone_spacing_m; self.speed_of_sound_mps=speed_of_sound_mps
    def localize(self,window:MultichannelAudioWindow)->AudioLocalization:
        try: import numpy as np
        except ImportError as exc: raise OptionalProviderDependency("GCC-PHAT requires NumPy") from exc
        a=np.asarray(window.channels[0],dtype=float); b=np.asarray(window.channels[1],dtype=float); n=a.size+b.size
        if a.size==0:return AudioLocalization(window.source_id,window.event_time,0.0,0.0)
        A=np.fft.rfft(a,n=n); B=np.fft.rfft(b,n=n); cross=A*np.conj(B); cross/=np.maximum(np.abs(cross),1e-12); corr=np.fft.irfft(cross,n=n); max_shift=min(int(window.sample_rate_hz*self.microphone_spacing_m/self.speed_of_sound_mps)+1,n//2)
        corr=np.concatenate((corr[-max_shift:],corr[:max_shift+1])); shift=int(np.argmax(np.abs(corr))-max_shift); delay=shift/window.sample_rate_hz; sin_angle=max(-1.0,min(1.0,delay*self.speed_of_sound_mps/self.microphone_spacing_m)); bearing=degrees(asin(sin_angle)); confidence=float(np.max(np.abs(corr))/(np.sum(np.abs(corr))+1e-12))
        return AudioLocalization(window.source_id,window.event_time,bearing,max(0.0,min(1.0,confidence)))

class AudioVisualAssociationProvider:
    provider_id="audio_visual_association"; provider_version="1"
    def __init__(self,*,maximum_bearing_error_degrees:float=20.0)->None:self.maximum_bearing_error_degrees=maximum_bearing_error_degrees
    def associate(self,localization:AudioLocalization,visual_bearings:Sequence[VisualBearing])->tuple[tuple[str,float],...]:
        out=[]
        for visual in visual_bearings:
            error=abs(((visual.bearing_degrees-localization.bearing_degrees+180)%360)-180)
            if error<=self.maximum_bearing_error_degrees:
                score=max(0.0,1-error/self.maximum_bearing_error_degrees)*min(localization.confidence,visual.confidence); out.append((visual.object_id,score))
        return tuple(sorted(out,key=lambda x:(-x[1],x[0])))
