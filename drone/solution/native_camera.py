"""Constraint-derived native-resolution survey for transferable data collection."""
import numpy as np
from dtos import RequestedViewDto

class NativeCamera:
    def __init__(self):self.cursor=0;self.steps=0;self.reset=False
    def choose(self,request):
        c=request.camera_constraints;v=request.view
        self.steps+=1
        if self.steps%22==0:self.reset=True
        if self.reset:
            if v.resolution_level==0:self.reset=False
            else:
                level=v.resolution_level-1
                b=c.bounds_for_level(level)
                if b is None:return None
                x=int(np.clip(v.center_x,b.minimum_center_x,b.maximum_center_x))
                y=int(np.clip(v.center_y,b.minimum_center_y,b.maximum_center_y))
                if level==0 or np.hypot(x-v.center_x,y-v.center_y)<=c.maximum_center_delta:
                    return RequestedViewDto(resolution_level=level,center_x=x,center_y=y)
                return None
        level=min(2,v.resolution_level+1)
        b=c.bounds_for_level(level)
        if b is None:return None
        xs=np.linspace(b.minimum_center_x,b.maximum_center_x,7)
        ys=np.linspace(b.minimum_center_y,b.maximum_center_y,5)
        targets=[(x,y) for row,y in enumerate(ys) for x in (xs if row%2==0 else xs[::-1])]
        target=np.array(targets[self.cursor%35]);current=np.array([v.center_x,v.center_y],float)
        delta=target-current;distance=np.linalg.norm(delta)
        if distance>c.maximum_center_delta:
            target=current+delta*(max(0,c.maximum_center_delta-2)/distance)
        elif level==2:self.cursor+=1
        x,y=np.rint(target).astype(int)
        x=int(np.clip(x,b.minimum_center_x,b.maximum_center_x));y=int(np.clip(y,b.minimum_center_y,b.maximum_center_y))
        if np.hypot(x-v.center_x,y-v.center_y)>c.maximum_center_delta:return None
        return RequestedViewDto(resolution_level=level,center_x=x,center_y=y)
