"""Single-worker, prewarmed official-protocol HTTP service."""
import argparse
import base64
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from common import ROOT, sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--imgsz",type=int,default=960)
    p.add_argument("--confidence",type=float,default=.02)
    p.add_argument("--model-family",choices=["yolo","rtdetr"],default="yolo")
    p.add_argument("--port",type=int,default=9053)
    p.add_argument("--log-dir",required=True)
    p.add_argument("--record",action="store_true")
    p.add_argument("--calibration",action="store_true")
    p.add_argument("--active-camera",action="store_true")
    p.add_argument("--native-survey",action="store_true")
    p.add_argument('--preserve-truncated',action='store_true')
    p.add_argument('--guard-camera',action='store_true')
    p.add_argument('--object-scout',action='store_true')
    p.add_argument('--flow-edge-camera',action='store_true')
    p.add_argument('--flow-edge-focus-camera',action='store_true')
    p.add_argument('--hybrid-detail-camera',action='store_true')
    p.add_argument('--static-level0',action='store_true')
    p.add_argument('--translation-median',type=int,default=0)
    p.add_argument('--rerank-head')
    p.add_argument('--rerank-alpha',type=float,default=1.)
    p.add_argument('--rerank-interval',type=int,default=1)
    p.add_argument('--confidence-decay',type=float,default=.96)
    p.add_argument('--observed-miss-decay',type=float,default=.5)
    p.add_argument('--min-track-confidence',type=float,default=.008)
    p.add_argument('--max-misses',type=int,default=2)
    p.add_argument('--dino-head')
    p.add_argument('--dino-threshold',type=float,default=.75)
    p.add_argument('--dino-margin',type=float,default=.25)
    p.add_argument('--dino-interval',type=int,default=1)
    p.add_argument('--dino-background-threshold',type=float,default=1.1)
    p.add_argument('--dino-background-margin',type=float,default=0.)
    p.add_argument('--rotation-interval',type=int,default=0)
    p.add_argument('--digital-tile',action='store_true')
    p.add_argument('--per-class-limit',type=int,default=0)
    p.add_argument('--max-output',type=int,default=0)
    p.add_argument('--min-hits',type=int,default=1)
    p.add_argument('--unconfirmed-scale',type=float,default=1.)
    p.add_argument('--confirmed-boost',type=float,default=1.)
    p.add_argument('--zero-motion-fallback',action='store_true')
    p.add_argument('--wide-interval',type=int,default=8)
    p.add_argument('--weak-observation-threshold',type=float,default=.025)
    p.add_argument('--weak-proposal-hits',type=int,default=2)
    p.add_argument('--near-duplicate-radius',type=float,default=0.)
    p.add_argument('--track-max-age',type=int,default=12)
    p.add_argument('--high-detail-birth-min-hits',type=int,default=1)
    p.add_argument('--association-radius',type=float,default=0.)
    p.add_argument('--teacher-weights')
    p.add_argument('--teacher-imgsz',type=int,default=960)
    p.add_argument('--teacher-interval',type=int,default=4)
    p.add_argument('--merge-teacher',action='store_true')
    p.add_argument('--teacher-class-only',action='store_true')
    p.add_argument('--teacher-unsupported-scale',type=float,default=1.)
    p.add_argument('--specialist-weights')
    p.add_argument('--specialist-imgsz',type=int,default=1280)
    p.add_argument('--acquisition-center',type=int,nargs=2,metavar=('X','Y'))
    args=p.parse_args()
    if args.teacher_class_only and not args.teacher_weights:p.error('--teacher-class-only requires --teacher-weights')
    if args.specialist_weights and not args.teacher_weights:p.error('--specialist-weights requires --teacher-weights')
    if args.merge_teacher and args.teacher_class_only:p.error('--merge-teacher and --teacher-class-only are mutually exclusive')
    if not 0 <= args.teacher_unsupported_scale <= 1:p.error('--teacher-unsupported-scale must be in [0,1]')
    if args.object_scout and args.native_survey:p.error('--object-scout and --native-survey are mutually exclusive')
    if args.object_scout and not args.guard_camera:p.error('--object-scout requires --guard-camera')
    if sum(bool(value) for value in [args.flow_edge_camera,args.flow_edge_focus_camera,args.hybrid_detail_camera,args.object_scout,args.native_survey])>1:p.error('camera policies are mutually exclusive')
    if args.acquisition_center and any([args.active_camera,args.native_survey,args.object_scout,args.flow_edge_camera,args.flow_edge_focus_camera,args.hybrid_detail_camera]):
        p.error('--acquisition-center is mutually exclusive with inference camera policies')
    if args.acquisition_center:
        from acquisition_camera import validate_native_center
        validate_native_center(*args.acquisition_center)
    os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/".cache/ultralytics"))
    from detector import Detector
    from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
    from fastapi import FastAPI,Request,HTTPException
    from service_ownership import add_identity,now
    import uvicorn
    logdir=Path(args.log_dir);logdir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    weights=Path(args.weights).resolve()
    manifest={"weights":str(weights.relative_to(ROOT)),"weights_sha256":sha256(weights),"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"source_sha256":{str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/"solution").glob("*.py")},"imgsz":args.imgsz,"confidence":args.confidence,"model_family":args.model_family,"policy":"full-view","pid":os.getpid(),"slurm_job_id":os.environ.get("SLURM_JOB_ID")}
    manifest["size_calibration"]=args.calibration
    if args.calibration:manifest["size_priors_sha256"]=sha256(ROOT/"solution/size_priors.json")
    detector_type=Detector
    extra={'model_family':args.model_family}
    if args.active_camera or args.native_survey or args.object_scout or args.flow_edge_camera or args.flow_edge_focus_camera or args.hybrid_detail_camera or args.static_level0:
        from active_detector import ActiveDetector
        detector_type=ActiveDetector
        extra['preserve_truncated']=args.preserve_truncated
        extra['guard_camera']=args.guard_camera
        extra['object_scout']=args.object_scout
        extra['flow_edge_camera']=args.flow_edge_camera
        extra['flow_edge_focus_camera']=args.flow_edge_focus_camera
        extra['hybrid_detail_camera']=args.hybrid_detail_camera
        extra['static_level0']=args.static_level0
        extra['translation_median']=args.translation_median
        extra['rerank_head']=args.rerank_head
        extra['rerank_alpha']=args.rerank_alpha
        extra['rerank_interval']=args.rerank_interval
        extra['confidence_decay']=args.confidence_decay
        extra['observed_miss_decay']=args.observed_miss_decay
        extra['min_track_confidence']=args.min_track_confidence
        extra['max_misses']=args.max_misses
        extra['dino_head']=args.dino_head
        extra['dino_threshold']=args.dino_threshold
        extra['dino_margin']=args.dino_margin
        extra['dino_interval']=args.dino_interval
        extra['dino_background_threshold']=args.dino_background_threshold
        extra['dino_background_margin']=args.dino_background_margin
        extra['rotation_interval']=args.rotation_interval
        extra['digital_tile']=args.digital_tile
        extra['per_class_limit']=args.per_class_limit
        extra['max_output']=args.max_output
        extra['min_hits']=args.min_hits
        extra['unconfirmed_scale']=args.unconfirmed_scale
        extra['confirmed_boost']=args.confirmed_boost
        extra['zero_motion_fallback']=args.zero_motion_fallback
        extra['wide_interval']=args.wide_interval
        extra['weak_observation_threshold']=args.weak_observation_threshold
        extra['weak_proposal_hits']=args.weak_proposal_hits
        extra['near_duplicate_radius']=args.near_duplicate_radius
        extra['track_max_age']=args.track_max_age
        extra['high_detail_birth_min_hits']=args.high_detail_birth_min_hits
        extra['association_radius']=args.association_radius
        extra['teacher_weights']=args.teacher_weights
        extra['teacher_imgsz']=args.teacher_imgsz
        extra['teacher_interval']=args.teacher_interval
        extra['merge_teacher']=args.merge_teacher
        extra['teacher_class_only']=args.teacher_class_only
        extra['teacher_unsupported_scale']=args.teacher_unsupported_scale
        extra['specialist_weights']=args.specialist_weights
        extra['specialist_imgsz']=args.specialist_imgsz
        manifest['static_level0']=args.static_level0
        manifest['translation_median']=args.translation_median
        manifest['rerank_head']=args.rerank_head
        manifest['rerank_alpha']=args.rerank_alpha
        manifest['rerank_interval']=args.rerank_interval
        manifest['confidence_decay']=args.confidence_decay
        manifest['observed_miss_decay']=args.observed_miss_decay
        manifest['min_track_confidence']=args.min_track_confidence
        manifest['max_misses']=args.max_misses
        manifest['guard_camera']=args.guard_camera
        manifest['preserve_truncated']=args.preserve_truncated
        manifest['dino_head']=args.dino_head
        manifest['rotation_interval']=args.rotation_interval
        manifest['digital_tile']=args.digital_tile
        manifest['per_class_limit']=args.per_class_limit
        manifest['max_output']=args.max_output
        manifest['min_hits']=args.min_hits
        manifest['unconfirmed_scale']=args.unconfirmed_scale
        manifest['confirmed_boost']=args.confirmed_boost
        manifest['zero_motion_fallback']=args.zero_motion_fallback
        manifest['wide_interval']=args.wide_interval
        manifest['weak_observation_threshold']=args.weak_observation_threshold
        manifest['weak_proposal_hits']=args.weak_proposal_hits
        manifest['near_duplicate_radius']=args.near_duplicate_radius
        manifest['track_max_age']=args.track_max_age
        manifest['high_detail_birth_min_hits']=args.high_detail_birth_min_hits
        manifest['association_radius']=args.association_radius
        manifest['flow_edge_camera']=args.flow_edge_camera
        manifest['flow_edge_focus_camera']=args.flow_edge_focus_camera
        manifest['hybrid_detail_camera']=args.hybrid_detail_camera
        manifest['teacher_weights']=args.teacher_weights
        if args.teacher_weights:
            manifest['teacher_weights_sha256']=sha256(Path(args.teacher_weights).resolve())
            manifest['teacher_imgsz']=args.teacher_imgsz
            manifest['teacher_interval']=args.teacher_interval
            manifest['merge_teacher']=args.merge_teacher
            manifest['teacher_class_only']=args.teacher_class_only
            manifest['teacher_unsupported_scale']=args.teacher_unsupported_scale
        manifest['specialist_weights']=args.specialist_weights
        if args.specialist_weights:
            manifest['specialist_weights_sha256']=sha256(Path(args.specialist_weights).resolve())
            manifest['specialist_imgsz']=args.specialist_imgsz
            manifest['specialist_target_classes']=['hangar','helicopter','jet_plane','large_tower']
        if args.dino_head:
            manifest['dino_head_sha256']=sha256(Path(args.dino_head).resolve())
            manifest['dino_threshold']=args.dino_threshold
            manifest['dino_margin']=args.dino_margin
            manifest['dino_interval']=args.dino_interval
            manifest['dino_background_threshold']=args.dino_background_threshold
            manifest['dino_background_margin']=args.dino_background_margin
        manifest["policy"]="visual-tracking-level1-coverage"
        if args.native_survey:
            extra["camera_depth"]=2
            manifest["policy"]="visual-tracking-native-survey"
        if args.object_scout:manifest['policy']='visual-tracking-object-scout'
        if args.flow_edge_camera:manifest['policy']='visual-tracking-flow-edge'
        if args.flow_edge_focus_camera:manifest['policy']='visual-tracking-flow-edge-focus'
        if args.hybrid_detail_camera:manifest['policy']='visual-tracking-hybrid-detail'
    detector=detector_type(weights,args.device,args.imgsz,args.confidence,args.calibration,**extra)
    if args.acquisition_center:
        manifest['policy']='public-native-grid-acquisition'
        manifest['acquisition_center']=list(args.acquisition_center)
        manifest['acquisition_only']=True
    if args.active_camera or args.native_survey or args.object_scout or args.flow_edge_camera or args.flow_edge_focus_camera or args.hybrid_detail_camera or args.static_level0:
        # Warm feature extraction, matching, geometry and camera transitions too.
        # Synthetic noise only: no reference or validation image is loaded here.
        import numpy as np
        from local_evaluator import Camera,build_request,render_view
        synthetic=np.random.default_rng(17).integers(0,256,(2160,3840,3),dtype=np.uint8)
        camera=Camera()
        for index in range(6):
            payload=build_request(index,index,camera,render_view(synthetic,camera),None)
            payload["sequence_id"]="synthetic-startup-warmup"
            response=detector.predict(DroneFlybyPredictRequestDto.model_validate(payload))
            if response.requested_view is not None:
                c=response.requested_view;camera.apply(c.resolution_level,c.center_x,c.center_y)
    add_identity(manifest)
    (logdir/"service_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("MODEL_READY "+json.dumps(manifest),flush=True)
    lock=threading.Lock()
    app=FastAPI()
    @app.get("/")
    def ready(nonce:str|None=None):
        if nonce is not None and (len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce)):raise HTTPException(status_code=400)
        receipt=dict(boot_id=manifest['boot_id'],ownership_digest=manifest['ownership_digest'],
                     weights_sha256=manifest['weights_sha256'],nonce=nonce,received_at=now(),ready=True,commit=manifest['commit'])
        if nonce is not None:
            with (logdir/'ownership_probes.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
        return receipt
    @app.post("/predict",response_model=DroneFlybyPredictResponseDto)
    async def predict(request:DroneFlybyPredictRequestDto,http_request:Request):
        start=time.perf_counter();error=None;received_at=now()
        with lock:
            try:
                if args.acquisition_center:
                    # Acquisition attempts collect pixels and are never detector
                    # candidates.  Skipping GPU inference removes avoidable frame
                    # loss while preserving the exact protocol and camera policy.
                    response=DroneFlybyPredictResponseDto(
                        request_id=request.request_id,frame=request.frame,annotations=[])
                else:
                    response=detector.predict(request)
            except Exception as exc:
                logging.exception("Inference failed")
                error=f"{type(exc).__name__}: {exc}"
                response=DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,annotations=[])
            if args.acquisition_center:
                from acquisition_camera import acquisition_view
                response.requested_view=acquisition_view(
                    request.view.resolution_level,*args.acquisition_center)
            elapsed=(time.perf_counter()-start)*1000
            row={"sequence_id":request.sequence_id,"frame":request.frame,"frame_index":request.frame_index,"request_id":request.request_id,"level":request.view.resolution_level,"elapsed_ms":elapsed,"error":error,"response":response.model_dump()}
            if args.specialist_weights:
                row['specialist_last_additions']=detector.specialist_last_additions
                row['specialist_additions_total']=detector.specialist_additions_total
            row.update(boot_id=manifest['boot_id'],ownership_digest=manifest['ownership_digest'],received_at=received_at,
                       finished_at=now(),route_path=http_request.url.path)
            if args.record:
                # Identifiers ONLY name audit files. They never influence predictions.
                seq=hashlib.sha256(request.sequence_id.encode()).hexdigest()[:16]
                folder=logdir/"recordings"/seq;folder.mkdir(parents=True,exist_ok=True)
                stem=f"{request.frame_index:06d}"
                # Offline and realtime local replays reuse identifiers. Keep
                # every observation and bind this response to its exact file.
                suffix=0
                while (folder/f"{stem}.png").exists() or (folder/f"{stem}.json").exists():
                    suffix+=1
                    stem=f"{request.frame_index:06d}_{suffix:03d}"
                (folder/f"{stem}.png").write_bytes(base64.b64decode(request.view.image))
                metadata=request.model_dump();metadata["view"].pop("image")
                (folder/f"{stem}.json").write_text(json.dumps(metadata))
                row["recording_path"]=str((folder/f"{stem}.png").relative_to(logdir))
            with open(logdir/"predictions.jsonl","a") as f:f.write(json.dumps(row)+"\n")
        return response
    uvicorn.run(app,host="0.0.0.0",port=args.port,workers=1,access_log=False)

if __name__=="__main__":main()
