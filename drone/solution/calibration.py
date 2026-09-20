"""Class-size calibration at fixed survey altitude; no scene positions."""
import json
from pathlib import Path
import numpy as np
from common import ROOT

def load_sizes():
    data=json.loads((ROOT/"solution/size_priors.json").read_text())
    return data["median_diagonal_source_pixels"]

def calibrate(annotations,width,height):
    sizes=load_sizes()
    for a in annotations:
        x1,y1,x2,y2=a.bbox
        diagonal=np.hypot((x2-x1)*width,(y2-y1)*height)
        # Broad octave-scale tolerance accommodates rotation and rendering changes.
        z=np.log(max(diagonal,1)/sizes[a.object_id])/np.log(2)
        factor=float(np.exp(-.5*z*z))
        a.confidence=float(a.confidence)*factor
    return [a for a in annotations if a.confidence>=.003]
