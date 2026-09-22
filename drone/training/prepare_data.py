"""Make annotated random crops and relocated-object compositions from Helsinki.

Validation frames are a same-instance diagnostic only. No inference component
loads these images, crops, annotations, or frame positions.
"""
import argparse
import json
from collections import defaultdict
import cv2
import numpy as np
import yaml
from common import ROOT, OFFICIAL, sha256
from dtos import OBJECT_CLASSES


def crop_view(image, annotations, region, size=(960, 540)):
    x,y,w,h=region
    result=cv2.resize(image[y:y+h,x:x+w],size,interpolation=cv2.INTER_AREA)
    boxes=[]
    for a in annotations:
        box=np.array(a["bbox"],dtype=float)-[x,y,x,y]
        box=np.clip(box,[0,0,0,0],[w,h,w,h])
        # Keep EVERY nonempty annotated fragment. No unlabelled negatives.
        if box[2]>box[0] and box[3]>box[1]:
            box*=np.array([size[0]/w,size[1]/h]*2)
            boxes.append((OBJECT_CLASSES.index(a["object_id"]),box))
    return result,boxes


def save_sample(out,split,index,image,boxes):
    stem=f"{index:05d}"
    cv2.imwrite(str(out/f"images/{split}/{stem}.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,97])
    h,w=image.shape[:2]
    rows=[]
    for cls,(x1,y1,x2,y2) in boxes:
        rows.append(f"{cls} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
    (out/f"labels/{split}/{stem}.txt").write_text("\n".join(rows)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--count",type=int,default=800)
    parser.add_argument("--seed",type=int,default=20260918)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    rng=np.random.default_rng(args.seed)
    out=ROOT/"solution/data/helsinki_aug"
    for kind in ["images","labels"]:
        for split in ["train","val"]:
            (out/kind/split).mkdir(parents=True,exist_ok=True)
    held_frames={4,9,14,19,24}
    scenes=[]
    patches=defaultdict(list)
    for p in sorted((OFFICIAL/"src/helsinki/annotations").glob("*.json")):
        doc=json.loads(p.read_text())
        img=cv2.imread(str(OFFICIAL/f"src/helsinki/images/{p.stem}.png"))
        scenes.append((doc,img))
        if doc["frame"] in held_frames:
            continue
        for a in doc["annotations"]:
            x1,y1,x2,y2=a["bbox"]
            if min(x1,y1)<12 or x2>3828 or y2>2148:
                continue
            # Tight labelled object plus 2px feathered border; all donors labelled.
            patch=img[y1-2:y2+2,x1-2:x2+2].copy()
            patches[OBJECT_CLASSES.index(a["object_id"])].append(patch)
    train=[s for s in scenes if s[0]["frame"] not in held_frames]
    assert len(patches)==16,"Every class needs a complete donor"
    counts=defaultdict(int)
    for idx in range(args.count):
        doc,img=train[int(rng.integers(len(train)))]
        # Train at all optical scales, plus intermediate scales.
        w=int(rng.choice([640,960,1280,1920,2560,3840]))
        h=w*9//16
        if rng.random()<.60:
            target=doc["annotations"][int(rng.integers(len(doc["annotations"])))]
            x1,y1,x2,y2=target["bbox"]
            x=int(np.clip((x1+x2)/2-rng.uniform(.15,.85)*w,0,3840-w))
            y=int(np.clip((y1+y2)/2-rng.uniform(.15,.85)*h,0,2160-h))
        else:
            x=int(rng.integers(3840-w+1)); y=int(rng.integers(2160-h+1))
        image,boxes=crop_view(img,doc["annotations"],(x,y,w,h))
        if idx%2==0:
            # Erase known objects before making synthetic backgrounds. The
            # organiser says every relevant object in these frames is annotated.
            mask=np.zeros(image.shape[:2],np.uint8)
            for _,(x1,y1,x2,y2) in boxes:
                cv2.rectangle(mask,(max(0,int(x1)-5),max(0,int(y1)-5)),(min(959,int(x2)+6),min(539,int(y2)+6)),255,-1)
            image=cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)
            boxes=[]
            for j in range(int(rng.integers(8,19))):
                cls=int(rng.integers(16))
                donor=patches[cls][int(rng.integers(len(patches[cls])))]
                # Full rotations change appearance and break the source layout.
                angle=float(rng.uniform(0,360))
                ph,pw=donor.shape[:2]
                matrix=cv2.getRotationMatrix2D((pw/2,ph/2),angle,1)
                side=int(np.ceil(np.hypot(ph,pw)))+4
                matrix[:,2]+=[side/2-pw/2,side/2-ph/2]
                alpha=np.ones((ph,pw),np.float32)
                alpha[[0,-1],:]=0;alpha[:,[0,-1]]=0
                alpha=cv2.GaussianBlur(alpha,(3,3),0)
                patch=cv2.warpAffine(donor,matrix,(side,side))
                alpha=cv2.warpAffine(alpha,matrix,(side,side))
                corners=np.array([[2,2],[pw-2,2],[pw-2,ph-2],[2,ph-2]],float)
                corners=corners@matrix[:,:2].T+matrix[:,2]
                bound=np.r_[corners.min(0),corners.max(0)]
                scale=float(rng.choice([.25,.5,1.0]))*float(rng.uniform(.65,1.4))
                target_size=max(8,int(side*scale))
                target_size=min(target_size,200)
                factor=target_size/side
                patch=cv2.resize(patch,(target_size,target_size),interpolation=cv2.INTER_AREA)
                alpha=cv2.resize(alpha,(target_size,target_size),interpolation=cv2.INTER_AREA)[...,None]
                px=int(rng.integers(960-target_size));py=int(rng.integers(540-target_size))
                b=bound*factor+[px,py,px,py]
                if any(min(b[2],b2[2])>max(b[0],b2[0])-3 and min(b[3],b2[3])>max(b[1],b2[1])-3 for _,b2 in boxes):
                    continue
                patch=np.clip(patch.astype(float)*rng.uniform(.7,1.3)+rng.uniform(-12,12),0,255)
                region=image[py:py+target_size,px:px+target_size]
                region[:]=(alpha*patch+(1-alpha)*region).astype(np.uint8)
                boxes.append((cls,b))
        for cls,_ in boxes: counts[OBJECT_CLASSES[cls]]+=1
        save_sample(out,"train",idx,image,boxes)
    val_index=0
    for doc,img in scenes:
        if doc["frame"] not in held_frames: continue
        image,boxes=crop_view(img,doc["annotations"],(0,0,3840,2160))
        save_sample(out,"val",val_index,image,boxes);val_index+=1
        for a in doc["annotations"]:
            x1,y1,x2,y2=a["bbox"]
            w=960;h=540
            x=int(np.clip((x1+x2)/2-w/2,0,3840-w));y=int(np.clip((y1+y2)/2-h/2,0,2160-h))
            image,boxes=crop_view(img,doc["annotations"],(x,y,w,h))
            save_sample(out,"val",val_index,image,boxes);val_index+=1
    (out/"data.yaml").write_text(yaml.safe_dump({"path":str(out),"train":"images/train","val":"images/val","names":dict(enumerate(OBJECT_CLASSES))}))
    manifest={"seed":args.seed,"training_images":args.count,"diagnostic_images":val_index,"train_frames":[d["frame"] for d,_ in train],"diagnostic_frames":sorted(held_frames),"split_warning":"same physical identities in both sets; NOT independent instance generalization","source":"official/drone-flyby/src/helsinki; supplied competition data, use permitted by organiser","class_counts":dict(counts),"background_policy":"all original annotations retained; synthetic backgrounds inpaint every annotated bbox plus margin","generator_sha256":sha256(__file__)}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (ROOT/"runs/2026-09-18-baseline/training_data_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
