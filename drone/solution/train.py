import argparse
import json
import os
import subprocess
from pathlib import Path
from common import ROOT, sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--name",default="2026-09-18-yolo11n-a")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--data",default="solution/data/helsinki_aug/data.yaml")
    p.add_argument("--initial",default="external/weights/yolo11n.pt")
    p.add_argument("--seed",type=int,default=20260918)
    p.add_argument("--freeze",type=int,default=0)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument('--architecture')
    args=p.parse_args()
    os.environ["YOLO_CONFIG_DIR"]=str(ROOT/".cache/ultralytics")
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(args.threads)
    # Custom heads are initialised before Ultralytics Trainer sets its seed.
    torch.manual_seed(args.seed)
    run=ROOT/"runs"/args.name
    checkpoint=run/"train/weights/last.pt" if args.resume else ROOT/args.initial
    model=YOLO(str(ROOT/args.architecture)).load(str(checkpoint)) if args.architecture and not args.resume else YOLO(str(checkpoint))
    architecture_audit=None
    if args.architecture and not args.resume:
        strides=model.model.stride.tolist()
        if 'p2' in args.architecture:assert strides==[4.,8.,16.,32.],strides
        model.model.to('cuda:0').eval()
        with torch.inference_mode():
            output=model.model(torch.zeros(1,3,256,256,device='cuda:0'))
            prediction=output[0] if isinstance(output,(tuple,list)) else output
            assert bool(torch.isfinite(prediction).all())
            architecture_audit=dict(strides=strides,initial_forward_shape=list(prediction.shape),initial_forward_finite=True,initialisation_seed=args.seed)
        del output,prediction
        model.model.cpu().train()
    options=dict(data=str(ROOT/"solution/data/helsinki_aug/data.yaml"),epochs=args.epochs,imgsz=960,batch=16,device=0,workers=4,project=str(run),name="train",exist_ok=True,seed=20260918,deterministic=True,optimizer="AdamW",lr0=.001,weight_decay=.001,cos_lr=True,patience=20,mosaic=.4,close_mosaic=8,degrees=180,translate=.15,scale=.5,fliplr=.5,flipud=.5,hsv_h=.035,hsv_s=.55,hsv_v=.45,mixup=0,cache=False,amp=False,plots=False,save_period=10)
    options["data"]=str(ROOT/args.data);options["seed"]=args.seed
    options["freeze"]=args.freeze;options["batch"]=args.batch
    options['workers']=args.workers
    options['lr0']=args.lr
    run.mkdir(parents=True,exist_ok=True)
    dataset_manifest=(ROOT/args.data).parent/'manifest.json'
    provenance={"options":options,"initial_weights_sha256":sha256(checkpoint),"hardware":torch.cuda.get_device_name(),"torch":torch.__version__,"commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"training_source_sha256":sha256(Path(__file__)),"dataset_yaml_sha256":sha256(ROOT/args.data),"dataset_manifest_sha256":sha256(dataset_manifest) if dataset_manifest.exists() else None}
    if args.architecture:provenance.update(architecture=args.architecture,architecture_sha256=sha256(ROOT/args.architecture),architecture_audit=architecture_audit)
    spec='resume_spec.json' if args.resume else 'training_spec.json'
    (run/spec).write_text(json.dumps(provenance,indent=2))
    if args.resume:model.train(resume=True)
    else:model.train(**options)
    weights=run/"train/weights/best.pt"
    (run/"weights.json").write_text(json.dumps({"path":str(weights.relative_to(ROOT)),"sha256":sha256(weights)},indent=2))

if __name__=="__main__":main()
