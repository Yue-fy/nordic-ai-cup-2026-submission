"""Transferable image-only detector. No reference images or scene map at inference."""
import time
import cv2
import numpy as np
from common import ROOT
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictResponseDto, RequestedViewDto
from utils import view_bbox_to_global, clip_bbox_to_frame, decode_view


def predictions_to_annotations(boxes,request):
    output=[]
    for x1,y1,x2,y2,confidence,cls in boxes:
        if not np.isfinite([x1,y1,x2,y2,confidence,cls]).all():continue
        bbox=clip_bbox_to_frame(view_bbox_to_global([x1/request.view.width,y1/request.view.height,x2/request.view.width,y2/request.view.height],request.view.source_region_xyxy,request.original_width,request.original_height))
        if bbox is None:continue
        output.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[int(cls)],bbox=list(bbox),confidence=float(np.clip(confidence,0,1))))
    return output


def full_view_command(request):
    if request.view.resolution_level==0:return None
    target=request.view.resolution_level-1
    bounds=request.camera_constraints.bounds_for_level(target)
    if bounds is None:return None
    x=int(np.clip(request.view.center_x,bounds.minimum_center_x,bounds.maximum_center_x))
    y=int(np.clip(request.view.center_y,bounds.minimum_center_y,bounds.maximum_center_y))
    distance=np.hypot(x-request.view.center_x,y-request.view.center_y)
    exempt=target==0 and request.camera_constraints.full_view_reset_exempt_from_delta
    if not exempt and distance>request.camera_constraints.maximum_center_delta:return None
    return RequestedViewDto(resolution_level=target,center_x=x,center_y=y)


class Detector:
    def __init__(self,weights,device="0",imgsz=960,confidence=.02,calibration=False,
                 model_family="yolo"):
        import torch
        from ultralytics import RTDETR,YOLO
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        self.device=device
        self.imgsz=imgsz
        self.confidence=confidence
        self.calibration=calibration
        if model_family not in {"yolo","rtdetr"}:raise ValueError("model_family must be yolo or rtdetr")
        self.model_family=model_family
        self.model=(RTDETR if model_family=="rtdetr" else YOLO)(str(weights))
        assert tuple(self.model.names[i] for i in range(16))==OBJECT_CLASSES,self.model.names
        self.model.fuse()
        # Empty warmup images may bypass NMS; initialise its CUDA path explicitly.
        from torchvision.ops import nms
        warm_device="cpu" if device=="cpu" else "cuda:"+str(device)
        nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]],device=warm_device),torch.tensor([.8,.7],device=warm_device),.5)
        for _ in range(4):self.infer(np.zeros((540,960,3),np.uint8))

    def infer(self,image):
        result=self.model.predict(image,imgsz=self.imgsz,device=self.device,half=self.device!="cpu",conf=self.confidence,iou=.5,agnostic_nms=True,max_det=100,verbose=False)[0]
        return result.boxes.data.cpu().numpy()

    def predict(self,request):
        image=decode_view(request.view)
        boxes=self.infer(image)
        annotations=predictions_to_annotations(boxes,request)
        if self.calibration:
            from calibration import calibrate
            annotations=calibrate(annotations,request.original_width,request.original_height)
        return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,annotations=annotations,requested_view=full_view_command(request))
