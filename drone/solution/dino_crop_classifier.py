"""Official DINOv2 frozen crop features for offline diagnostics."""
import cv2
import numpy as np
import torch
from torch import nn
from common import ROOT

MEAN=np.array([.485,.456,.406],np.float32);STD=np.array([.229,.224,.225],np.float32)


def crop_tensor(image,box,padding=.3,size=224):
    box=np.asarray(box,float);centre=(box[:2]+box[2:])/2
    side=max(4.,float(max(box[2:]-box[:2]))*(1+2*padding));scale=size/side
    matrix=np.array([[scale,0,size/2-scale*centre[0]],[0,scale,size/2-scale*centre[1]]],np.float32)
    crop=cv2.warpAffine(image,matrix,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    rgb=crop[:,:,::-1].astype(np.float32)/255
    return torch.from_numpy(((rgb-MEAN)/STD).transpose(2,0,1).copy())


def feature_model():
    repo=ROOT/'external/models/dinov2';weights=ROOT/'external/weights/dinov2_vits14_pretrain.pth'
    model=torch.hub.load(str(repo),'dinov2_vits14',source='local',weights=str(weights))
    return model.requires_grad_(False).eval()


class DinoCropClassifier:
    def __init__(self,checkpoint,device='cuda:0'):
        data=torch.load(checkpoint,map_location='cpu',weights_only=True);self.device=device
        self.model=feature_model().to(device);self.head=nn.Linear(384,17).to(device)
        self.head.load_state_dict(data['head']);self.head.eval()

    @torch.inference_mode()
    def probabilities(self,image,boxes):
        if not len(boxes):return np.empty((0,17),np.float32)
        patches=torch.stack([crop_tensor(image,b[:4]) for b in boxes]).to(self.device);outputs=[]
        for batch in patches.split(64):
            with torch.autocast('cuda',enabled=str(self.device).startswith('cuda')):
                outputs.append(self.head(self.model(batch)).float().softmax(1).cpu())
        return torch.cat(outputs).numpy()


class DinoRerankHead:
    """Frozen-DINOv2 object-vs-clutter score used only to re-rank detections.

    It never deletes a box: COCO AP is insensitive to low-ranked false
    positives, and deleting small objects is the documented failure mode of
    earlier crop filters. Appearance only; no position or frame number.
    """

    def __init__(self, checkpoint, device='cuda:0'):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.device = device
        self.model = feature_model().to(device)
        self.head = nn.Linear(384, 2).to(device)
        self.head.load_state_dict(data['head'])
        self.head.eval()

    @torch.inference_mode()
    def object_probability(self, image, boxes):
        if not len(boxes):
            return np.empty((0,), np.float32)
        patches = torch.stack([crop_tensor(image, b[:4]) for b in boxes]).to(self.device)
        out = []
        for batch in patches.split(64):
            with torch.autocast('cuda', enabled=str(self.device).startswith('cuda')):
                out.append(self.head(self.model(batch)).float().softmax(1)[:, 1].cpu())
        return torch.cat(out).numpy()
