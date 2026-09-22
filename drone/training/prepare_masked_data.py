"""Masked-object training with explicitly reviewed target-free background crops."""
import json
import argparse
import shutil
from collections import defaultdict
import cv2
import numpy as np
import yaml
from common import ROOT,sha256
from dtos import OBJECT_CLASSES
from prepare_data import save_sample

def main():
    p=argparse.ArgumentParser();p.add_argument('--multiview',action='store_true');p.add_argument('--refined',action='store_true');p.add_argument('--safe-appearance',action='store_true');p.add_argument('--official-boxes',action='store_true');args=p.parse_args()
    assert sum([args.multiview,args.refined,args.safe_appearance])<=1
    assert not (args.official_boxes and (args.multiview or args.refined or args.safe_appearance)), 'Public foreground boxes are manual tight boxes, not authoritative official annotations'
    cv2.setNumThreads(2)
    seed=20261010 if args.safe_appearance else (20260923 if args.refined else (20260921 if args.multiview else 20260919))
    rng=np.random.default_rng(seed)
    out=ROOT/"solution/data"/('official_box_aug' if args.official_boxes else ('safe_appearance_aug' if args.safe_appearance else ('refined_aug' if args.refined else ('multiview_aug' if args.multiview else 'masked_aug'))))
    for kind in ["images","labels"]:
        for split in ["train","val"]:(out/kind/split).mkdir(parents=True,exist_ok=True)
    donor_dir=ROOT/"solution/data/reviewed_donors"
    records=json.loads((donor_dir/"manifest.json").read_text())
    donors=defaultdict(list)
    for row in records:
        assert row["reviewed"]
        im=cv2.imread(str(donor_dir/row["file"]),cv2.IMREAD_UNCHANGED)
        if args.official_boxes:
            x1,y1,x2,y2=row['source_bbox']
            assert im.shape[:2]==(y2-y1,x2-x1), 'Official donor canvas must preserve the original annotation box'
        donors[OBJECT_CLASSES.index(row["class"])].append(im)
    additional=defaultdict(list)
    if args.multiview or args.refined or args.safe_appearance:
        folder=ROOT/'solution/data/public_donors_reviewed'
        public_records=json.loads((folder/'manifest.json').read_text())
        for row in public_records:
            assert row['reviewed']
            if row['split']=='train':
                additional[OBJECT_CLASSES.index(row['class_name'])].append(cv2.imread(str(folder/row['donor_file']),cv2.IMREAD_UNCHANGED))
        if args.refined or args.safe_appearance:
            folder_v2=ROOT/'solution/data/public_donors_reviewed_v2'
            records_v2=json.loads((folder_v2/'manifest.json').read_text())
            for row in records_v2:
                assert row['reviewed'] and row['split']=='train'
                additional[OBJECT_CLASSES.index(row['class_name'])].append(cv2.imread(str(folder_v2/row['donor_file']),cv2.IMREAD_UNCHANGED))
            public_records+=records_v2
    hard_negatives=[]
    if args.refined or args.safe_appearance:
        hard_records=json.loads((ROOT/'external/reviewed_hard_negatives_manifest.json').read_text())['rows']
        for row in hard_records:
            assert row['review_status']=='visually_reviewed_empty_for_official_16_classes'
            hard_negatives.append(cv2.imread(str(ROOT/row['file'])))
    background_dir=ROOT/"external/data/reviewed_backgrounds"
    bg_records=[r for r in json.loads((background_dir/"manifest.json").read_text()) if r["review_status"]=="reviewed_empty_for_official_16_classes"]
    backgrounds=[cv2.imread(str(background_dir/r["file"])) for r in bg_records]
    assert len(donors)==16 and len(backgrounds)>=5
    counts=defaultdict(int)
    synthetic_count=1600 if (args.refined or args.safe_appearance) else 1200
    for idx in range(synthetic_count):
        # Each patch has been reviewed. Unreviewed images never become negatives.
        rows=[]
        for _ in range(2):
            tiles=[]
            for _ in range(2):
                tile=backgrounds[int(rng.integers(len(backgrounds)))]
                if rng.random()<.5:tile=cv2.flip(tile,1)
                tiles.append(cv2.resize(tile,(480,270)))
            rows.append(np.concatenate(tiles,axis=1))
        image=np.concatenate(rows,axis=0)
        if args.refined or args.safe_appearance:
            # Every hard-negative crop was inspected in full; unknown regions
            # are never treated as target-free. Keep the crops at varied scales.
            for _ in range(int(rng.integers(2,6))):
                patch=hard_negatives[int(rng.integers(len(hard_negatives)))];ph,pw=patch.shape[:2]
                scale=float(rng.uniform(.6,2.5));w=max(6,min(300,round(pw*scale)));h=max(6,min(260,round(ph*scale)))
                patch=cv2.resize(patch,(w,h))
                if rng.random()<.5:patch=cv2.flip(patch,1)
                x=int(rng.integers(960-w));y=int(rng.integers(540-h));image[y:y+h,x:x+w]=patch
        if args.safe_appearance:
            # Relight only already reviewed target-free background pixels.
            gamma=float(rng.uniform(.65,1.45))
            image=np.clip((image.astype(np.float32)/255.)**gamma*255,0,255)
            image*=rng.uniform(.82,1.18,(1,1,3))
            haze=float(rng.uniform(0,.12));image=image*(1-haze)+rng.uniform(150,235)*haze
            image=np.clip(image,0,255).astype(np.uint8)
        boxes=[]
        official_boxes=[]
        if idx%8!=0 and idx<1200:
            for j in range(int(rng.integers(8,21))):
                cls=int(rng.integers(16))
                pool=additional[cls] if (args.multiview or args.refined or args.safe_appearance) and additional[cls] and rng.random()<.5 else donors[cls]
                donor=pool[int(rng.integers(len(pool)))]
                if args.multiview:
                    # Foreground-only projective variation changes overhead shape
                    # without preserving locations, routes or scene background.
                    dh,dw=donor.shape[:2]
                    donor=cv2.resize(donor,(max(4,round(dw*rng.uniform(.7,1.3))),max(4,round(dh*rng.uniform(.7,1.3)))))
                    dh,dw=donor.shape[:2]
                    src=np.float32([[0,0],[dw-1,0],[dw-1,dh-1],[0,dh-1]])
                    dst=src+rng.uniform(-.12,.12,(4,2))*[dw,dh]
                    donor=cv2.warpPerspective(donor,cv2.getPerspectiveTransform(src,dst.astype(np.float32)),(dw,dh))
                ph,pw=donor.shape[:2]
                side=int(np.ceil(np.hypot(ph,pw)))+8
                transform=cv2.getRotationMatrix2D((pw/2,ph/2),float(rng.uniform(0,360)),1)
                transform[:,2]+=[side/2-pw/2,side/2-ph/2]
                rgba=cv2.warpAffine(donor,transform,(side,side))
                # Use one shared optical scale per composition, mild object variation.
                scale=float([.25,.5,1.0][idx%3])*float(rng.uniform(.8,1.3))
                size=max(8,min(220,round(side*scale)))
                rgba=cv2.resize(rgba,(size,size),interpolation=cv2.INTER_AREA)
                alpha=rgba[:,:,3:4].astype(np.float32)/255
                ys,xs=np.where(alpha[:,:,0]>.1)
                if len(xs)<3:continue
                x=int(rng.integers(960-size));y=int(rng.integers(540-size))
                box=np.array([xs.min()+x,ys.min()+y,xs.max()+x+1,ys.max()+y+1],float)
                if any(min(box[2],b[2])>max(box[0],b[0])-3 and min(box[3],b[3])>max(box[1],b[1])-3 for _,b in boxes):continue
                foreground=np.clip(rgba[:,:,:3].astype(float)*rng.uniform(.65,1.4)+rng.uniform(-15,15),0,255)
                # Mild colour channel changes and blur vary material/rendering.
                channel_range=(.65,1.35) if args.multiview else (.85,1.15)
                foreground=np.clip(foreground*rng.uniform(*channel_range,(1,1,3)),0,255)
                if args.multiview and rng.random()<.25:
                    # Material colour is not guaranteed by the organiser.
                    # Preserve luminance texture, vary base albedo independently.
                    luminance=foreground.mean(2,keepdims=True)
                    centred=luminance-luminance[alpha>.1].mean()
                    albedo=rng.uniform(30,210,(1,1,3))
                    foreground=np.clip(albedo+centred*rng.uniform(.55,1.25),0,255)
                patch=image[y:y+size,x:x+size]
                if args.safe_appearance:
                    angle=float(rng.uniform(0,2*np.pi));length=float(rng.uniform(.08,.55)*size)
                    dx,dy=np.cos(angle)*length,np.sin(angle)*length
                    shadow=cv2.warpAffine(alpha[:,:,0],np.float32([[1,0,dx],[0,1,dy]]),(size,size),flags=cv2.INTER_LINEAR,borderValue=0)
                    sigma=max(.6,float(rng.uniform(.015,.08)*size))
                    shadow=cv2.GaussianBlur(shadow,(0,0),sigma)
                    opacity=float(rng.uniform(.12,.48))
                    patch[:]=np.clip(patch.astype(np.float32)*(1-opacity*shadow[...,None]),0,255).astype(np.uint8)
                patch[:]=(alpha*foreground+(1-alpha)*patch).astype(np.uint8)
                boxes.append((cls,box));counts[OBJECT_CLASSES[cls]]+=1
                if args.official_boxes:
                    corners=np.array([[0,0,1],[pw,0,1],[pw,ph,1],[0,ph,1]],float)@transform.T
                    corners*=size/side
                    authoritative=np.r_[corners.min(0),corners.max(0)]+[x,y,x,y]
                    assert 0<=authoritative[0]<authoritative[2]<=960 and 0<=authoritative[1]<authoritative[3]<=540
                    official_boxes.append((cls,authoritative))
        # Preserve the old tight-mask collision checks and RNG calls exactly,
        # so the label-only variant produces byte-identical training images.
        save_sample(out,"train",idx,image,official_boxes if args.official_boxes else boxes)
    # Retain all 400 natural, fully annotated multiscale crops from A.
    original=ROOT/"solution/data/helsinki_aug"
    for idx in range(1,800,2):
        stem=f"{idx:05d}"
        for kind,ext in [("images","jpg"),("labels","txt")]:
            shutil.copyfile(original/f"{kind}/train/{stem}.{ext}",out/f"{kind}/train/natural_{stem}.{ext}")
    for kind in ["images","labels"]:
        for file in (original/kind/"val").iterdir():
            if file.suffix in {".jpg",".txt"}:shutil.copyfile(file,out/kind/"val"/file.name)
    (out/"data.yaml").write_text(yaml.safe_dump({"path":str(out),"train":"images/train","val":"images/val","names":dict(enumerate(OBJECT_CLASSES))}))
    manifest={"seed":seed,"synthetic_images":synthetic_count,"natural_images":400,"reviewed_background_crops":bg_records,"donor_manifest_sha256":sha256(donor_dir/"manifest.json"),"per_class_instances":dict(counts),"validation_scope":"same-instance Helsinki; public sequence now development data because reviewed negative crops are used","false_negative_prevention":"Only visually reviewed empty crops used as backgrounds; every pasted object labelled; natural crops preserve original annotations"}
    if args.multiview or args.refined or args.safe_appearance:
        manifest.update(public_donors_manifest_sha256=sha256(folder/'manifest.json'),public_train_instances=sorted({r['instance'] for r in public_records if r['split']=='train'}),heldout_instances=sorted({r['instance'] for r in public_records if r['split']=='holdout'}),material_recolour_probability=.25 if args.multiview else 0,generalisation_limit='Appearance augmentation does not guarantee unseen 3D-model variants')
    if args.refined or args.safe_appearance:manifest.update(public_donors_v2_sha256=sha256(folder_v2/'manifest.json'),hard_negative_manifest_sha256=sha256(ROOT/'external/reviewed_hard_negatives_manifest.json'),hard_negative_crops=len(hard_negatives),additional_empty_composites=400)
    if args.safe_appearance:manifest.update(safe_appearance=dict(background_gamma=[.65,1.45],background_channel_gain=[.82,1.18],haze=[0,.12],shadow_length_fraction=[.08,.55],shadow_opacity=[.12,.48],foreground_pixels_and_boxes_unchanged=True))
    if args.official_boxes:
        baseline=ROOT/'solution/data/masked_aug'
        image_files=sorted((out/'images/train').glob('*.jpg'))
        assert len(image_files)==1600
        for f in image_files:
            assert sha256(f)==sha256(baseline/'images/train'/f.name), f'Image changed in label-only control: {f.name}'
        for kind in ['images','labels']:
            for f in (out/kind/'val').iterdir():
                assert sha256(f)==sha256(baseline/kind/'val'/f.name)
        manifest.update(label_definition='Transform original official bounding-box canvas corners; mask controls pixels only',
            image_equivalence=dict(verified_byte_identical_train_images=len(image_files),baseline='solution/data/masked_aug',validation_images_and_labels_identical=True),
            public_foreground_donors_used=False)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    run=ROOT/"runs"/('2026-09-18-official-box-labels' if args.official_boxes else ('2026-09-19-safe-appearance-data' if args.safe_appearance else ('2026-09-18-yolo11s-refined' if args.refined else ('2026-09-18-yolo11n-multiview' if args.multiview else '2026-09-18-yolo11n-masked'))));run.mkdir(parents=True,exist_ok=True)
    (run/"data_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"images":synthetic_count+400,"counts":dict(counts)}))

if __name__=="__main__":main()
