"""Online visual motion tracking and geometry-only camera coverage.

State is made exclusively from this sequence's received images and detections.
No training/reference image, validation route or answer map is loaded.
"""
from dataclasses import dataclass, field
import cv2
import numpy as np
from common import ROOT
from detector import Detector, predictions_to_annotations
from calibration import calibrate
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictResponseDto, RequestedViewDto
from utils import decode_view


def source_points(points,region):
    x,y,x2,y2=region
    return np.asarray(points,dtype=np.float32)*[(x2-x)/960,(y2-y)/540]+[x,y]


class VisualMotion:
    def __init__(self,fallback_window=1,zero_fallback=False,translation_median=0):
        if fallback_window<1:raise ValueError('fallback_window must be positive')
        if translation_median<0:raise ValueError('translation_median cannot be negative')
        self.orb=cv2.ORB_create(nfeatures=1600,fastThreshold=12)
        self.previous=None
        self.last_translation=np.zeros(2)
        self.fallback_window=fallback_window
        self.zero_fallback=zero_fallback
        # A survey drone flies a straight line over static ground, so the
        # per-frame ground translation is close to constant. Smoothing the
        # measured translation over a window removes the per-frame estimation
        # noise that otherwise accumulates into track drift. Geometry only:
        # the window is learned from this sequence's own images.
        self.translation_median=translation_median
        self.accepted_translations=[]
        self.translation_history=[]

    def update(self,image,region,gap):
        keypoints,descriptors=self.orb.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
        points=source_points([k.pt for k in keypoints],region) if keypoints else np.empty((0,2))
        matrix=np.array([[1.,0.,0.],[0.,1.,0.]])
        quality=0
        if self.previous is not None and descriptors is not None:
            prev_points,prev_desc=self.previous
            if prev_desc is not None and len(prev_desc)>12 and len(descriptors)>12:
                matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(prev_desc,descriptors,k=2)
                good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.72*pair[1].distance]
                if len(good)>=12:
                    src=np.float32([prev_points[m.queryIdx] for m in good])
                    dst=np.float32([points[m.trainIdx] for m in good])
                    candidate,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=12,maxIters=1500,confidence=.995)
                    if candidate is not None:
                        scale=np.hypot(candidate[0,0],candidate[1,0])
                        quality=int(inliers.sum())
                        displacement=candidate@np.array([1920,1080,1])-np.array([1920,1080])
                        if quality>=12 and quality/len(good)>.30 and .92<scale<1.08 and np.linalg.norm(displacement)<500*max(gap,1):
                            matrix=candidate
                            self.last_translation=displacement/max(gap,1)
                            self.translation_history.append(self.last_translation.copy())
                            self.translation_history=self.translation_history[-self.fallback_window:]
                        else:quality=0
        if quality==0 and self.previous is not None:
            if not self.zero_fallback:
                velocity=np.median(self.translation_history,axis=0) if self.translation_history else self.last_translation
                matrix[:,2]=velocity*gap
        if self.translation_median:
            if quality>0:
                self.accepted_translations.append(self.last_translation.copy())
                self.accepted_translations=self.accepted_translations[-self.translation_median:]
            if self.accepted_translations:
                velocity=np.median(self.accepted_translations,axis=0)
                matrix=np.array([[1.,0.,velocity[0]*gap],[0.,1.,velocity[1]*gap]])
        self.previous=(points,descriptors)
        return matrix,quality


@dataclass
class Track:
    box:np.ndarray
    evidence:np.ndarray
    confidence:float
    age:int=0
    misses:int=0
    hits:int=1
    birth_level:int=1


def overlap(a,b):
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=float(np.prod(wh))
    return inter/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-inter+1e-6)


def overlap_matrix(a,b):
    """Same scalar IoU formula, evaluated for a complete pairwise batch."""
    a=np.asarray(a,dtype=float).reshape(-1,4)
    b=np.asarray(b,dtype=float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=np.prod(wh,axis=2)
    areas_a=np.prod(a[:,2:]-a[:,:2],axis=1)
    areas_b=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/(areas_a[:,None]+areas_b[None,:]-inter+1e-6)


def suppress_near_duplicates(annotations,width,height,radius=0.,max_area_ratio=4.):
    """Keep confidence order while merging only nearby, similarly sized same-class boxes."""
    if radius<=0 or len(annotations)<2:return annotations
    kept=[]
    scale=np.array([width,height,width,height],dtype=float)
    for annotation in sorted(annotations,key=lambda a:-a.confidence):
        box=np.asarray(annotation.bbox,dtype=float)*scale
        centre=(box[:2]+box[2:])/2
        diagonal=float(np.linalg.norm(box[2:]-box[:2]))
        area=float(np.prod(np.maximum(0,box[2:]-box[:2])))
        duplicate=False
        for other,other_box,other_centre,other_diagonal,other_area in kept:
            if other.object_id!=annotation.object_id:continue
            ratio=max(area,other_area)/max(min(area,other_area),1e-6)
            distance=float(np.linalg.norm(centre-other_centre))
            if ratio<=max_area_ratio and distance<=radius*max(diagonal,other_diagonal):
                duplicate=True;break
        if not duplicate:kept.append((annotation,box,centre,diagonal,area))
    return [row[0] for row in kept]


def inverse_rot90_boxes(boxes,original_width):
    """Map boxes from a 90-degree counter-clockwise view to the source view."""
    boxes=np.asarray(boxes).copy()
    if not len(boxes):return boxes.reshape(-1,6)
    x1,y1,x2,y2=boxes[:,:4].T
    boxes[:,:4]=np.stack([original_width-y2,x1,original_width-y1,x2],axis=1)
    return boxes


def merge_detector_boxes(primary,secondary,iou_threshold=.5,max_det=100):
    """Merge two already-NMSed detector outputs without score calibration."""
    import torch
    from torchvision.ops import nms
    primary=np.asarray(primary).reshape(-1,6)
    secondary=np.asarray(secondary).reshape(-1,6)
    if not len(primary):return secondary[:max_det]
    if not len(secondary):return primary[:max_det]
    boxes=np.concatenate([primary,secondary])
    keep=nms(torch.as_tensor(boxes[:,:4]),torch.as_tensor(boxes[:,4]),iou_threshold)
    return boxes[keep.cpu().numpy()[:max_det]]


def add_consensus_specialist_proposals(primary,specialist,verifier,target_classes=(0,1,2,4),
                                       iou_threshold=.5,confidence_threshold=.25,max_det=100):
    """Add only new specialist boxes independently confirmed by the verifier."""
    import torch
    from torchvision.ops import nms
    primary=np.asarray(primary).copy().reshape(-1,6)
    specialist=np.asarray(specialist).reshape(-1,6)
    verifier=np.asarray(verifier).reshape(-1,6)
    if not len(specialist) or not len(verifier):return primary
    primary_overlap=overlap_matrix(specialist[:,:4],primary[:,:4]) if len(primary) else np.zeros((len(specialist),0))
    verifier_overlap=overlap_matrix(specialist[:,:4],verifier[:,:4])
    additions=[];targets=set(int(value) for value in target_classes)
    for index,row in enumerate(specialist):
        cls=int(row[5])
        if cls not in targets or row[4]<confidence_threshold:continue
        if len(primary) and float(primary_overlap[index].max())>=iou_threshold:continue
        verifier_index=int(np.argmax(verifier_overlap[index]))
        support=verifier[verifier_index]
        if (verifier_overlap[index,verifier_index]<iou_threshold or
                support[4]<confidence_threshold or int(support[5])!=cls):continue
        accepted=row.copy();accepted[4]=min(row[4],support[4]);additions.append(accepted)
    if not additions:return primary
    boxes=np.concatenate([primary,np.asarray(additions)]) if len(primary) else np.asarray(additions)
    keep=nms(torch.as_tensor(boxes[:,:4]),torch.as_tensor(boxes[:,4]),iou_threshold)
    return boxes[keep.cpu().numpy()[:max_det]]


def fuse_teacher_classes(primary,secondary,iou_threshold=.5,
                         teacher_confidence=.25,confidence_margin=.05,
                         unsupported_scale=1.):
    """Use teacher evidence and optionally downweight unsupported primary boxes."""
    if not 0<=unsupported_scale<=1:raise ValueError('unsupported_scale must be in [0,1]')
    primary=np.asarray(primary).copy().reshape(-1,6)
    secondary=np.asarray(secondary).reshape(-1,6)
    if not len(primary):return primary
    if not len(secondary):
        primary[:,4]*=unsupported_scale
        return primary
    left_top=np.maximum(primary[:,None,:2],secondary[None,:,:2])
    right_bottom=np.minimum(primary[:,None,2:4],secondary[None,:,2:4])
    intersection=np.maximum(0,right_bottom-left_top).prod(2)
    primary_area=np.maximum(0,primary[:,2:4]-primary[:,:2]).prod(1)
    secondary_area=np.maximum(0,secondary[:,2:4]-secondary[:,:2]).prod(1)
    overlaps=intersection/np.maximum(primary_area[:,None]+secondary_area[None,:]-intersection,1e-9)
    matches=overlaps.argmax(1)
    for index,teacher_index in enumerate(matches):
        teacher=secondary[teacher_index]
        supported=(overlaps[index,teacher_index]>=iou_threshold and
                   teacher[4]>=teacher_confidence)
        if not supported:
            primary[index,4]*=unsupported_scale
        elif teacher[4]>=primary[index,4]+confidence_margin:
            primary[index,5]=teacher[5]
    return primary


def refine_dino_boxes(boxes,probabilities,class_threshold,class_margin,
                      background_threshold=1.1,background_margin=0.):
    """Apply a frozen crop head without changing geometry or detector scores."""
    boxes=np.asarray(boxes).copy()
    if not len(boxes):return boxes.reshape(-1,6)
    keep=np.ones(len(boxes),dtype=bool)
    for i,(box,p) in enumerate(zip(boxes,probabilities)):
        object_probability=float(np.max(p[:16]))
        if p[16]>=background_threshold and p[16]-object_probability>=background_margin:
            keep[i]=False
            continue
        original=int(box[5]);candidate=int(np.argmax(p[:16]))
        if (candidate!=original and p[candidate]>=class_threshold and
                p[candidate]-p[original]>=class_margin and p[16]<.5):
            box[5]=candidate
    return boxes[keep]


class OnlineTracker:
    def __init__(self,preserve_truncated=False,max_age=12,evidence_decay=.97,confidence_decay=.96,
                 missing_motion_decay=.85,observed_miss_decay=.5,max_misses=2,min_confidence=.008,
                 per_class_limit=0,max_output=0,min_hits=1,unconfirmed_scale=1.,confirmed_boost=1.,
                 weak_observation_threshold=.025,weak_proposal_hits=2,near_duplicate_radius=0.,
                 high_detail_birth_min_hits=1,association_radius=0.):
        self.tracks=[]
        self.weak_tracks=[]
        self.preserve_truncated=preserve_truncated
        self.max_age=max_age
        self.evidence_decay=evidence_decay
        self.confidence_decay=confidence_decay
        self.missing_motion_decay=missing_motion_decay
        self.observed_miss_decay=observed_miss_decay
        self.max_misses=max_misses
        self.min_confidence=min_confidence
        if per_class_limit<0:raise ValueError('per_class_limit cannot be negative')
        self.per_class_limit=per_class_limit
        if max_output<0:raise ValueError('max_output cannot be negative')
        self.max_output=max_output
        if min_hits<1:raise ValueError('min_hits must be positive')
        self.min_hits=min_hits
        if not 0<=unconfirmed_scale<=1:raise ValueError('unconfirmed_scale must be in [0,1]')
        self.unconfirmed_scale=unconfirmed_scale
        if confirmed_boost<1:raise ValueError('confirmed_boost must be at least 1')
        self.confirmed_boost=confirmed_boost
        if not 0<=weak_observation_threshold<=.025:
            raise ValueError('weak_observation_threshold must be in [0,.025]')
        if weak_proposal_hits<2:raise ValueError('weak_proposal_hits must be at least 2')
        self.weak_observation_threshold=weak_observation_threshold
        self.weak_proposal_hits=weak_proposal_hits
        if near_duplicate_radius<0:raise ValueError('near_duplicate_radius cannot be negative')
        self.near_duplicate_radius=near_duplicate_radius
        if high_detail_birth_min_hits<1:raise ValueError('high_detail_birth_min_hits must be positive')
        self.high_detail_birth_min_hits=high_detail_birth_min_hits
        if association_radius<0:raise ValueError('association_radius cannot be negative')
        self.association_radius=association_radius

    def update(self,annotations,matrix,gap,region,width,height,motion_quality,observation_level=1,
               class_evidence=None):
        if class_evidence is not None and len(class_evidence)!=len(annotations):
            raise ValueError('class_evidence must align with annotations')
        scaling=np.hypot(matrix[0,0],matrix[1,0])
        for t in self.tracks:
            centre=(t.box[:2]+t.box[2:])/2
            half=(t.box[2:]-t.box[:2])*scaling/2
            centre=matrix@np.r_[centre,1]
            t.box=np.r_[centre-half,centre+half]
            t.age+=gap;t.evidence*=self.evidence_decay**gap;t.confidence*=self.confidence_decay**gap
            if motion_quality==0:t.confidence*=self.missing_motion_decay
        for t in self.weak_tracks:
            centre=(t.box[:2]+t.box[2:])/2
            half=(t.box[2:]-t.box[:2])*scaling/2
            centre=matrix@np.r_[centre,1]
            t.box=np.r_[centre-half,centre+half]
            t.age+=gap;t.evidence*=self.evidence_decay**gap;t.confidence*=self.confidence_decay**gap
            if motion_quality==0:t.confidence*=self.missing_motion_decay
        matched=set()
        evidence_rows=class_evidence if class_evidence is not None else [None]*len(annotations)
        observed=[pair for pair in sorted(zip(annotations,evidence_rows),
                                          key=lambda pair:-pair[0].confidence)
                  if pair[0].confidence>=.025]
        observed_annotations=[pair[0] for pair in observed]
        boxes=np.array([a.bbox for a in observed_annotations],dtype=float).reshape(-1,4)*[width,height,width,height]
        # New tracks are marked matched immediately, so they were never eligible
        # for association later in the same update. Only pre-existing tracks
        # need to participate in this matrix, preserving the original behaviour.
        association=overlap_matrix(boxes,[t.box for t in self.tracks])
        available=np.ones(len(self.tracks),dtype=bool)
        existing_tracks=self.tracks[:len(available)]
        for row,((a,class_row),box) in enumerate(zip(observed,boxes)):
            values=np.where(available,association[row],-1.)
            if len(values):
                # Python max((iou,index)) preferred the larger index on ties.
                best_index=len(values)-1-int(np.argmax(values[::-1]))
                best=(values[best_index],best_index)
            else:best=(0,-1)
            idx=OBJECT_CLASSES.index(a.object_id)
            if best[0]<=.20 and self.association_radius>0 and len(values):
                centres=np.array([(t.box[:2]+t.box[2:])/2 for t in existing_tracks])
                centre=(box[:2]+box[2:])/2
                distances=np.linalg.norm(centres-centre,axis=1)
                track_diagonals=np.array([np.linalg.norm(t.box[2:]-t.box[:2]) for t in existing_tracks])
                diagonal=np.linalg.norm(box[2:]-box[:2])
                normalized=distances/np.maximum(np.maximum(track_diagonals,diagonal),1e-6)
                track_areas=np.array([np.prod(np.maximum(t.box[2:]-t.box[:2],0)) for t in existing_tracks])
                area=np.prod(np.maximum(box[2:]-box[:2],0))
                ratios=np.maximum(track_areas,area)/np.maximum(np.minimum(track_areas,area),1e-6)
                track_classes=np.array([int(np.argmax(t.evidence)) for t in existing_tracks])
                eligible=available&(track_classes==idx)&(ratios<=4)&(normalized<=self.association_radius)
                if np.any(eligible):
                    fallback_values=np.where(eligible,normalized,np.inf)
                    best=(.201,int(np.argmin(fallback_values)))
            if class_row is None:
                observation_evidence=np.zeros(16);observation_evidence[idx]=a.confidence
            else:
                observation_evidence=np.asarray(class_row,dtype=float).copy()
                if observation_evidence.shape!=(16,) or not np.isfinite(observation_evidence).all() or np.any(observation_evidence<0):
                    raise ValueError('each class_evidence row must contain 16 finite non-negative values')
                total=observation_evidence.sum()
                if total<=0:
                    observation_evidence[idx]=1.;total=1.
                observation_evidence*=a.confidence/total
            if best[0]>.20:
                t=self.tracks[best[1]];matched.add(best[1])
                available[best[1]]=False
                if self.preserve_truncated:
                    merged=box.copy();margin=3*(region[2]-region[0])/960
                    if region[0]>0 and box[0]<=region[0]+margin:merged[0]=min(merged[0],t.box[0])
                    if region[1]>0 and box[1]<=region[1]+margin:merged[1]=min(merged[1],t.box[1])
                    if region[2]<width and box[2]>=region[2]-margin:merged[2]=max(merged[2],t.box[2])
                    if region[3]<height and box[3]>=region[3]-margin:merged[3]=max(merged[3],t.box[3])
                    t.box=merged
                else:t.box=box
                t.evidence+=observation_evidence
                t.confidence=max(t.confidence,a.confidence)
                t.age=0;t.misses=0;t.hits+=1
            else:
                self.tracks.append(Track(box,observation_evidence,a.confidence,birth_level=observation_level));matched.add(len(self.tracks)-1)
        for i,t in enumerate(self.tracks):
            centre=(t.box[:2]+t.box[2:])/2
            if i not in matched and region[0]<centre[0]<region[2] and region[1]<centre[1]<region[3]:
                t.misses+=1
                if t.misses>=self.max_misses:t.confidence*=self.observed_miss_decay
        self.tracks=[t for t in self.tracks if t.age<=self.max_age and t.confidence>self.min_confidence and t.box[2]>0 and t.box[3]>0 and t.box[0]<width and t.box[1]<height]
        self.tracks=sorted(self.tracks,key=lambda t:-t.confidence)[:250]

        # Keep weak observations in a separate bank so they cannot change the
        # established >=.025 association, evidence, geometry or confidence.
        if self.weak_observation_threshold<.025:
            weak_observed=[pair for pair in sorted(zip(annotations,evidence_rows),
                                                   key=lambda pair:-pair[0].confidence)
                           if self.weak_observation_threshold<=pair[0].confidence<.025]
            weak_boxes=np.array([a.bbox for a,_ in weak_observed],dtype=float).reshape(-1,4)*[width,height,width,height]
            if len(weak_boxes) and self.tracks:
                clear=np.max(overlap_matrix(weak_boxes,[t.box for t in self.tracks]),axis=1)<=.5
                weak_observed=[pair for pair,keep in zip(weak_observed,clear) if keep]
                weak_boxes=weak_boxes[clear]
            weak_association=overlap_matrix(weak_boxes,[t.box for t in self.weak_tracks])
            weak_available=np.ones(len(self.weak_tracks),dtype=bool)
            weak_matched=set()
            for row,((a,class_row),box) in enumerate(zip(weak_observed,weak_boxes)):
                values=np.where(weak_available,weak_association[row],-1.)
                if len(values):
                    best_index=len(values)-1-int(np.argmax(values[::-1]))
                    best=(values[best_index],best_index)
                else:best=(0,-1)
                idx=OBJECT_CLASSES.index(a.object_id)
                if class_row is None:
                    observation_evidence=np.zeros(16);observation_evidence[idx]=a.confidence
                else:
                    observation_evidence=np.asarray(class_row,dtype=float).copy()
                    if observation_evidence.shape!=(16,) or not np.isfinite(observation_evidence).all() or np.any(observation_evidence<0):
                        raise ValueError('each class_evidence row must contain 16 finite non-negative values')
                    total=observation_evidence.sum()
                    if total<=0:
                        observation_evidence[idx]=1.;total=1.
                    observation_evidence*=a.confidence/total
                if best[0]>.20:
                    t=self.weak_tracks[best[1]];weak_matched.add(best[1]);weak_available[best[1]]=False
                    t.box=box;t.evidence+=observation_evidence;t.confidence=max(t.confidence,a.confidence)
                    t.age=0;t.misses=0;t.hits+=1
                else:
                    self.weak_tracks.append(Track(box,observation_evidence,a.confidence))
                    weak_matched.add(len(self.weak_tracks)-1)
            for i,t in enumerate(self.weak_tracks):
                centre=(t.box[:2]+t.box[2:])/2
                if i not in weak_matched and region[0]<centre[0]<region[2] and region[1]<centre[1]<region[3]:
                    t.misses+=1
                    if t.misses>=self.max_misses:t.confidence*=self.observed_miss_decay
            weak_floor=max(.0005,self.weak_observation_threshold*.1)
            self.weak_tracks=[t for t in self.weak_tracks if t.age<=self.max_age and
                              t.confidence>weak_floor and t.box[2]>0 and t.box[3]>0 and
                              t.box[0]<width and t.box[1]<height]
            self.weak_tracks=sorted(self.weak_tracks,key=lambda t:-t.confidence)[:250]
        output=[]
        suppression=overlap_matrix([t.box for t in self.tracks],[t.box for t in self.tracks])>.5
        suppressed=np.zeros(len(self.tracks),dtype=bool)
        for i,t in enumerate(self.tracks):
            if suppressed[i]:continue
            if t.hits<self.min_hits:continue
            if t.birth_level>=2 and t.hits<self.high_detail_birth_min_hits:continue
            box=np.clip(t.box/[width,height,width,height],0,1)
            if box[2]<=box[0] or box[3]<=box[1]:continue
            suppressed|=suppression[i]
            cls=int(np.argmax(t.evidence))
            certainty=t.evidence[cls]/max(t.evidence.sum(),1e-9)
            scale=self.unconfirmed_scale if t.hits<2 else self.confirmed_boost
            confidence=min(1.,t.confidence*certainty*scale)
            output.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[cls],bbox=box.tolist(),confidence=float(confidence)))
        if self.weak_observation_threshold<.025 and self.weak_tracks:
            strong_boxes=[t.box for t in self.tracks]
            weak_suppression=overlap_matrix([t.box for t in self.weak_tracks],
                                            [t.box for t in self.weak_tracks])>.5
            weak_suppressed=np.zeros(len(self.weak_tracks),dtype=bool)
            for i,t in enumerate(self.weak_tracks):
                if weak_suppressed[i] or t.hits<self.weak_proposal_hits:continue
                if strong_boxes and np.max(overlap_matrix([t.box],strong_boxes))>.5:continue
                box=np.clip(t.box/[width,height,width,height],0,1)
                if box[2]<=box[0] or box[3]<=box[1]:continue
                weak_suppressed|=weak_suppression[i]
                cls=int(np.argmax(t.evidence))
                certainty=t.evidence[cls]/max(t.evidence.sum(),1e-9)
                # A weak proposal must remain below the main-track boundary.
                confidence=min(np.nextafter(.025,0.),t.confidence*certainty)
                output.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[cls],bbox=box.tolist(),confidence=float(confidence)))
        output=suppress_near_duplicates(output,width,height,self.near_duplicate_radius)
        if self.max_output and len(output)>self.max_output:
            # Bound the response payload. COCO AP is insensitive to
            # low-ranked detections, but a four-times larger response body
            # measurably costs frames on a bandwidth-limited return path,
            # and a frame that never arrives is scored as empty.
            output=sorted(output,key=lambda a:-a.confidence)[:self.max_output]
        if self.per_class_limit:
            counts={}
            limited=[]
            for annotation in sorted(output,key=lambda a:-a.confidence):
                count=counts.get(annotation.object_id,0)
                if count<self.per_class_limit:
                    limited.append(annotation)
                    counts[annotation.object_id]=count+1
            output=limited
        return output


class CoverageCamera:
    def __init__(self,wide_interval=8):
        if wide_interval<2:raise ValueError('wide_interval must be at least 2')
        self.cursor=0;self.steps=0;self.wide_interval=wide_interval
    def choose(self,request):
        c=request.camera_constraints;v=request.view
        self.steps+=1
        # Periodic wide observations reset drift and search for new objects.
        if self.steps%self.wide_interval==0 and 0 in c.allowed_resolution_levels:
            b=c.bounds_for_level(0)
            return RequestedViewDto(resolution_level=0,center_x=int(b.minimum_center_x),center_y=int(b.minimum_center_y))
        b=c.bounds_for_level(1)
        if b is None:return None
        lo,hi=b.minimum_center_x,b.maximum_center_x
        top,bottom=b.minimum_center_y,b.maximum_center_y
        middle=(lo+hi)//2
        targets=[(lo,top),(middle,top),(hi,top),(hi,bottom),(middle,bottom),(lo,bottom)]
        target=np.array(targets[self.cursor%6],float)
        current=np.array([v.center_x,v.center_y],float)
        delta=target-current;length=np.linalg.norm(delta)
        if length>c.maximum_center_delta:
            target=current+delta*(max(c.maximum_center_delta-2,0)/length)
        else:self.cursor+=1
        x,y=np.rint(target).astype(int)
        x=int(np.clip(x,lo,hi));y=int(np.clip(y,top,bottom))
        if np.hypot(x-v.center_x,y-v.center_y)>c.maximum_center_delta:return None
        return RequestedViewDto(resolution_level=1,center_x=x,center_y=y)


class StaticWideCamera:
    """Hold the full-frame Level-0 view every frame.

    Geometry only: it never reads a scene position, frame number or map.
    """
    def choose(self,request):
        c=request.camera_constraints
        if request.view.resolution_level==0:return None
        if 0 not in c.allowed_resolution_levels:return None
        b=c.bounds_for_level(0)
        if b is None:return None
        return RequestedViewDto(resolution_level=0,center_x=int(b.minimum_center_x),center_y=int(b.minimum_center_y))


class ActiveDetector(Detector):
    def __init__(self,*args,camera_depth=1,preserve_truncated=False,guard_camera=False,object_scout=False,
                 flow_edge_camera=False,
                 flow_edge_focus_camera=False,hybrid_detail_camera=False,static_level0=False,translation_median=0,max_output=0,
                 rerank_head=None,rerank_alpha=1.,rerank_interval=1,
                 confidence_decay=.96,observed_miss_decay=.5,min_track_confidence=.008,max_misses=2,
                 dino_head=None,dino_threshold=.75,dino_margin=.25,dino_interval=1,rotation_interval=0,
                 digital_tile=False,
                 per_class_limit=0,dino_background_threshold=1.1,dino_background_margin=0.,
                 teacher_weights=None,teacher_imgsz=960,teacher_interval=4,merge_teacher=False,
                 teacher_class_only=False,teacher_unsupported_scale=1.,min_hits=1,
                 specialist_weights=None,specialist_imgsz=1280,
                 unconfirmed_scale=1.,confirmed_boost=1.,zero_motion_fallback=False,wide_interval=8,
                 weak_observation_threshold=.025,weak_proposal_hits=2,near_duplicate_radius=0.,
                 track_max_age=12,high_detail_birth_min_hits=1,association_radius=0.,**kwargs):
        # Detector.__init__ deliberately calls self.infer for CUDA warmup.
        # Define this before super() so that virtual dispatch stays valid.
        self.dino=None
        self.confidence_decay=confidence_decay
        self.observed_miss_decay=observed_miss_decay
        self.min_track_confidence=min_track_confidence
        self.max_misses=max_misses
        self.rerank=None
        self.rerank_alpha=float(rerank_alpha)
        if rerank_interval<1:raise ValueError('rerank_interval must be positive')
        self.rerank_interval=rerank_interval
        self.rerank_enabled_for_frame=True
        self.rotation_enabled=False
        self.rotation_interval=rotation_interval
        self.digital_tile=digital_tile
        self.tile_enabled_for_frame=False
        self.tile_index=0
        self.per_class_limit=per_class_limit
        self.max_output=max_output
        self.min_hits=min_hits
        self.unconfirmed_scale=unconfirmed_scale
        self.confirmed_boost=confirmed_boost
        self.zero_motion_fallback=zero_motion_fallback
        self.wide_interval=wide_interval
        self.weak_observation_threshold=weak_observation_threshold
        self.weak_proposal_hits=weak_proposal_hits
        self.near_duplicate_radius=near_duplicate_radius
        if track_max_age<1:raise ValueError('track_max_age must be positive')
        self.track_max_age=track_max_age
        self.high_detail_birth_min_hits=high_detail_birth_min_hits
        self.association_radius=association_radius
        self.teacher=None
        self.specialist=None
        self.specialist_last_additions=0
        self.specialist_additions_total=0
        self.merge_teacher=merge_teacher
        self.teacher_class_only=teacher_class_only
        self.teacher_unsupported_scale=teacher_unsupported_scale
        self.teacher_enabled_for_frame=False
        super().__init__(*args,**kwargs)
        self.camera_depth=camera_depth
        self.preserve_truncated=preserve_truncated
        self.guard_camera=guard_camera
        self.object_scout=object_scout
        self.flow_edge_camera=flow_edge_camera
        self.flow_edge_focus_camera=flow_edge_focus_camera
        self.hybrid_detail_camera=hybrid_detail_camera
        self.static_level0=static_level0
        self.translation_median=translation_median
        self.command_guard=None
        self.sequence=None
        self.last_index=None
        self.motion=None
        self.tracker=None
        self.camera=None
        self.dino_threshold=dino_threshold
        self.dino_margin=dino_margin
        self.dino_interval=dino_interval
        self.dino_background_threshold=dino_background_threshold
        self.dino_background_margin=dino_background_margin
        self.dino_enabled_for_frame=True
        if dino_interval<1:raise ValueError('dino_interval must be positive')
        if rotation_interval<0:raise ValueError('rotation_interval cannot be negative')
        if teacher_interval<1:raise ValueError('teacher_interval must be positive')
        self.teacher_interval=teacher_interval
        if teacher_weights is not None:
            self.teacher=Detector(teacher_weights,self.device,teacher_imgsz,self.confidence,False)
        if specialist_weights is not None:
            if self.teacher is None:raise ValueError('specialist requires a verifier teacher')
            self.specialist=Detector(specialist_weights,self.device,specialist_imgsz,self.confidence,False)
        if rerank_head is not None:
            from dino_crop_classifier import DinoRerankHead
            device='cpu' if str(self.device)=='cpu' else 'cuda:'+str(self.device)
            self.rerank=DinoRerankHead(rerank_head,device)
            self.rerank.object_probability(np.zeros((540,960,3),np.uint8),np.array([[350,150,610,390,1,0]],float))
        if dino_head is not None:
            from dino_crop_classifier import DinoCropClassifier
            device='cpu' if str(self.device)=='cpu' else 'cuda:'+str(self.device)
            self.dino=DinoCropClassifier(dino_head,device)
            # Keep first-request latency out of the real-time sequence.
            self.dino.probabilities(np.zeros((540,960,3),np.uint8),np.array([[350,150,610,390,1,0]],float))

    def infer(self,image):
        self.specialist_last_additions=0
        if self.teacher_enabled_for_frame:
            teacher_boxes=self.teacher.infer(image)
            primary_boxes=super().infer(image)
            if self.teacher_class_only:
                boxes=fuse_teacher_classes(primary_boxes,teacher_boxes,
                                           unsupported_scale=self.teacher_unsupported_scale)
            elif self.merge_teacher:boxes=merge_detector_boxes(primary_boxes,teacher_boxes)
            else:boxes=teacher_boxes
            if self.specialist is not None:
                specialist_boxes=self.specialist.infer(image)
                previous_count=len(boxes)
                boxes=add_consensus_specialist_proposals(boxes,specialist_boxes,teacher_boxes)
                self.specialist_last_additions=max(0,len(boxes)-previous_count)
                self.specialist_additions_total+=self.specialist_last_additions
        else:boxes=super().infer(image)
        if self.digital_tile and self.tile_enabled_for_frame:
            h,w=image.shape[:2];half_w=w//2;half_h=h//2
            regions=[(0,0,half_w,half_h),(half_w,0,w,half_h),
                     (0,half_h,half_w,h),(half_w,half_h,w,h)]
            left,top,right,bottom=regions[self.tile_index%4]
            extra=super().infer(image[top:bottom,left:right])
            if len(extra):
                extra=extra.copy();crop_w=right-left;crop_h=bottom-top
                keep=(extra[:,0]>3)&(extra[:,1]>3)&(extra[:,2]<crop_w-3)&(extra[:,3]<crop_h-3)
                extra=extra[keep]
                extra[:,:4]+=[left,top,left,top]
                boxes=merge_detector_boxes(boxes,extra)
        if self.rotation_enabled:
            import torch
            from torchvision.ops import nms
            rotated=np.ascontiguousarray(np.rot90(image,1))
            extra=inverse_rot90_boxes(super().infer(rotated),image.shape[1])
            if len(extra):
                boxes=np.concatenate([boxes,extra]) if len(boxes) else extra
                keep=nms(torch.as_tensor(boxes[:,:4]),torch.as_tensor(boxes[:,4]),.5).cpu().numpy()[:100]
                boxes=boxes[keep]
        if self.rerank is not None and self.rerank_enabled_for_frame and len(boxes):
            probability=self.rerank.object_probability(image,boxes)
            boxes=np.asarray(boxes).copy()
            boxes[:,4]=boxes[:,4]*np.power(np.clip(probability,1e-4,1.),self.rerank_alpha)
        if self.dino is None or not self.dino_enabled_for_frame or not len(boxes):return boxes
        probabilities=self.dino.probabilities(image,boxes)
        return refine_dino_boxes(boxes,probabilities,self.dino_threshold,self.dino_margin,
                                 self.dino_background_threshold,self.dino_background_margin)

    def predict(self,request):
        if request.sequence_id!=self.sequence or self.last_index is None or request.frame_index<=self.last_index:
            self.sequence=request.sequence_id
            self.specialist_last_additions=0;self.specialist_additions_total=0
            self.motion=VisualMotion(zero_fallback=self.zero_motion_fallback,translation_median=self.translation_median);self.tracker=OnlineTracker(self.preserve_truncated,max_age=self.track_max_age,confidence_decay=self.confidence_decay,observed_miss_decay=self.observed_miss_decay,min_confidence=self.min_track_confidence,max_misses=self.max_misses,per_class_limit=self.per_class_limit,max_output=self.max_output,min_hits=self.min_hits,unconfirmed_scale=self.unconfirmed_scale,confirmed_boost=self.confirmed_boost,weak_observation_threshold=self.weak_observation_threshold,weak_proposal_hits=self.weak_proposal_hits,near_duplicate_radius=self.near_duplicate_radius,high_detail_birth_min_hits=self.high_detail_birth_min_hits,association_radius=self.association_radius);self.camera=CoverageCamera(self.wide_interval)
            if self.guard_camera:
                from camera_guard import CameraCommandGuard
                self.command_guard=CameraCommandGuard()
            if self.camera_depth==2:
                from native_camera import NativeCamera
                self.camera=NativeCamera()
            if self.object_scout:
                from scout_camera import ScoutCamera
                self.camera=ScoutCamera()
            if self.flow_edge_camera or self.flow_edge_focus_camera:
                from flow_edge_camera import FlowEdgeCamera
                self.camera=FlowEdgeCamera(self.wide_interval,focus_after_wide=self.flow_edge_focus_camera)
            if self.hybrid_detail_camera:
                from hybrid_detail_camera import HybridDetailCamera
                self.camera=HybridDetailCamera(wide_interval=self.wide_interval)
            if self.static_level0:
                self.camera=StaticWideCamera()
            self.last_index=request.frame_index-1
        gap=request.frame_index-self.last_index;self.last_index=request.frame_index
        image=decode_view(request.view)
        matrix,quality=self.motion.update(image,request.view.source_region_xyxy,gap)
        # The tracker keeps accumulated class evidence on intervening frames.
        # Sparse frozen-feature updates preserve that memory at lower latency.
        self.dino_enabled_for_frame=request.frame_index%self.dino_interval==0
        self.rerank_enabled_for_frame=request.frame_index%self.rerank_interval==0
        self.rotation_enabled=self.rotation_interval>0 and request.frame_index%self.rotation_interval==0
        self.tile_enabled_for_frame=self.digital_tile
        self.tile_index=request.frame_index%4
        self.teacher_enabled_for_frame=self.teacher is not None and request.frame_index%self.teacher_interval==0
        boxes=self.infer(image)
        current_annotations=predictions_to_annotations(boxes,request)
        current_annotations=calibrate(current_annotations,request.original_width,request.original_height)
        annotations=self.tracker.update(current_annotations,matrix,gap,request.view.source_region_xyxy,request.original_width,request.original_height,quality,request.view.resolution_level)
        if self.object_scout or self.hybrid_detail_camera:
            command=self.camera.choose(request,self.tracker.tracks)
        elif self.flow_edge_camera or self.flow_edge_focus_camera:
            command=self.camera.choose(request,self.motion.last_translation,quality,current_annotations)
        else:
            command=self.camera.choose(request)
        if self.command_guard is not None:
            command=self.command_guard.filter(request,command)
        return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,annotations=annotations,requested_view=command)
