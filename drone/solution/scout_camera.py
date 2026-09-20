"""Object-directed, bounded zoom cycles using current online tracks only."""
import numpy as np
from dtos import RequestedViewDto


class ScoutCamera:
    def __init__(self):
        self.tick=0;self.wide_age=0;self.cycles=0;self.target=None
        self.returning=False;self.examined={};self.hits=0;self.exploring=False
        self.explore_cursor=0

    @staticmethod
    def command(request,level,centre):
        c=request.camera_constraints;v=request.view
        if level not in c.allowed_resolution_levels:return None
        b=c.bounds_for_level(level)
        if b is None:return None
        target=np.clip(centre,[b.minimum_center_x,b.minimum_center_y],[b.maximum_center_x,b.maximum_center_y]).astype(float)
        current=np.array([v.center_x,v.center_y],float);delta=target-current;distance=np.linalg.norm(delta)
        exempt=level==0 and c.full_view_reset_exempt_from_delta
        if not exempt and distance>c.maximum_center_delta:
            target=current+delta*(max(0,c.maximum_center_delta-2)/distance)
        target=np.rint(target).astype(int)
        target=np.clip(target,[b.minimum_center_x,b.minimum_center_y],[b.maximum_center_x,b.maximum_center_y])
        if not exempt and np.linalg.norm(target-current)>c.maximum_center_delta:return None
        return RequestedViewDto(resolution_level=level,center_x=int(target[0]),center_y=int(target[1]))

    @staticmethod
    def contains(request,box):
        region=np.array(request.view.source_region_xyxy,float)
        margin=.03*(region[2:]-region[:2])
        return bool(np.all(box[:2]>=region[:2]+margin) and np.all(box[2:]<=region[2:]-margin))

    def choose(self,request,tracks):
        self.tick+=1;level=request.view.resolution_level
        self.wide_age=0 if level==0 else self.wide_age+1
        alive={id(t):t for t in tracks}
        self.examined={k:v for k,v in self.examined.items() if k in alive and self.tick-v<30}
        if self.target is not None and id(self.target) not in alive:
            self.target=None;self.returning=True
        if self.returning and level==0:
            self.returning=False;self.target=None;self.exploring=False;self.hits=0
        if self.wide_age>=6:self.returning=True
        if not self.returning and self.target is None and not self.exploring:
            eligible=[]
            for t in tracks:
                side=t.box[2:]-t.box[:2];centre=(t.box[:2]+t.box[2:])/2
                if t.age>1 or t.confidence<.12 or id(t) in self.examined:continue
                if np.any(side<4) or max(side)>800 or not (0<=centre[0]<request.original_width and 0<=centre[1]<request.original_height):continue
                certainty=float(t.evidence.max()/max(t.evidence.sum(),1e-9))
                priority=t.confidence*(1+min(2.,48/max(min(side),1)))*(1+.5*(1-certainty))
                eligible.append((priority,t))
            # A geometry-only exploration cycle follows every two directed ones.
            self.exploring=not eligible or self.cycles%3==2
            self.cycles+=1
            if not self.exploring:self.target=max(eligible,key=lambda pair:pair[0])[1]
        if not self.returning:
            if self.exploring:
                if level==1:self.hits+=1
                if self.hits>=2:
                    self.returning=True;self.explore_cursor+=1
                else:
                    b=request.camera_constraints.bounds_for_level(1)
                    if b is None:return None
                    xs=[b.minimum_center_x,(b.minimum_center_x+b.maximum_center_x)//2,b.maximum_center_x]
                    ys=[b.minimum_center_y,b.maximum_center_y]
                    x=xs[self.explore_cursor%3];y=ys[(self.explore_cursor//3)%2]
                    return self.command(request,1,[x,y])
            elif self.target is not None:
                t=self.target;side=t.box[2:]-t.box[:2]
                certainty=float(t.evidence.max()/max(t.evidence.sum(),1e-9))
                desired=2 if min(side)<96 or certainty<.7 or t.confidence<.55 else 1
                if level>=desired and self.contains(request,t.box):self.hits+=1
                if self.hits>=(2 if desired==2 else 1):
                    self.examined[id(t)]=self.tick;self.returning=True
                else:
                    destination=level+int(np.sign(desired-level))
                    return self.command(request,destination,(t.box[:2]+t.box[2:])/2)
        if self.returning:
            if self.target is not None:self.examined[id(self.target)]=self.tick
            destination=max(0,level-1)
            return self.command(request,destination,[request.view.center_x,request.view.center_y])
        return None
