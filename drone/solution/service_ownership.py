import os
"""Bind a public score to this workspace's exact running service instance.

Boot markers distinguish service instances at the standard /predict address.
Never scan unrelated endpoints or retain foreign replies.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid
from common import ROOT

PUBLIC_ORIGIN=os.environ.get('DRONE_PUBLIC_ORIGIN','http://127.0.0.1:9053')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def add_identity(manifest):
    manifest['boot_id']=uuid.uuid4().hex
    manifest['prediction_path']='/predict'
    manifest['identity_path']='/'
    manifest['ownership_digest']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return manifest


def public_url(manifest):
    boot=manifest['boot_id']
    if len(boot)!=32 or any(c not in '0123456789abcdef' for c in boot):raise ValueError('Invalid local boot identity')
    if manifest['prediction_path']!='/predict':raise ValueError('Expected standard prediction path')
    return PUBLIC_ORIGIN+manifest['prediction_path']


def check_reply(reply,manifest,nonce):
    expected=dict(boot_id=manifest['boot_id'],ownership_digest=manifest['ownership_digest'],
                  weights_sha256=manifest['weights_sha256'],nonce=nonce)
    return isinstance(reply,dict) and all(reply.get(k)==v for k,v in expected.items())


def probe(run,phase,origin=PUBLIC_ORIGIN):
    manifest=json.loads((run/'service/service_manifest.json').read_text())
    public_url(manifest)  # Validate expected path before contacting anything.
    nonce=uuid.uuid4().hex
    record=dict(checked_at=now(),phase=phase,boot_id=manifest['boot_id'],ownership_digest=manifest['ownership_digest'],matched=False)
    try:
        url=origin+manifest['identity_path']+'?nonce='+nonce
        request=urllib.request.Request(url,headers={'Cache-Control':'no-store','Accept':'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=3) as response:
            raw=response.read(4097)
            if response.status!=200 or len(raw)>4096:raise ValueError('Unexpected identity response')
        if not check_reply(json.loads(raw),manifest,nonce):raise ValueError('Service identity mismatch')
        # Corroborate the fresh network challenge in our own service's local log.
        receipts=run/'service/ownership_probes.jsonl'
        if not receipts.exists():raise ValueError('Missing local identity receipt')
        recent=receipts.read_text().splitlines()[-20:]
        if not any((r:=json.loads(line)).get('nonce')==nonce and r.get('boot_id')==manifest['boot_id'] for line in recent):
            raise ValueError('Challenge not handled by local owned service')
        record['matched']=True
    except Exception as exc:
        # Do not print/store another service's body, weights, code, or headers.
        record['failure_type']=type(exc).__name__
    record['finished_at']=now()
    with (run/'ownership_checks.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    if not record['matched']:raise RuntimeError('Public route ownership was not proven; see sanitized ownership_checks.jsonl')
    return record


def timestamp(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def audit_window(records,start,end,boot,digest,max_gap=8):
    """Need owned successes bracketing the attempt without an unexplained gap.

    A transport timeout returns no identity and is therefore a missing sample,
    not evidence that another service answered.  It is safe to ignore only when
    successful nonce challenges around it still satisfy the same maximum gap.
    """
    matching=[r for r in records if r.get('boot_id')==boot and r.get('ownership_digest')==digest]
    successful=[r for r in matching if r.get('matched') is True]
    before=[r for r in successful if timestamp(r['checked_at'])<=start]
    after=[r for r in successful if timestamp(r['checked_at'])>=end]
    if not before or not after:return dict(passed=False,reason='missing_interval_bracket')
    lo=max(timestamp(r['checked_at']) for r in before);hi=min(timestamp(r['checked_at']) for r in after)
    chosen=sorted((r for r in successful if lo<=timestamp(r['checked_at'])<=hi),key=lambda r:r['checked_at'])
    failures=[r for r in matching if lo<=timestamp(r['checked_at'])<=hi and not r.get('matched')]
    unsafe=[r for r in failures if r.get('failure_type') not in {'URLError','TimeoutError'}]
    gaps=[timestamp(b['checked_at'])-timestamp(a['checked_at']) for a,b in zip(chosen,chosen[1:])]
    passed=not unsafe and max(gaps,default=0)<=max_gap and start-lo<=max_gap and hi-end<=max_gap
    return dict(passed=passed,checks=len(chosen),maximum_gap_seconds=max(gaps,default=0),allowed_gap_seconds=max_gap,
                tolerated_transport_failures=len(failures)-len(unsafe),unsafe_failures=len(unsafe),
                reason=None if passed else 'identity_failure_or_unobserved_interval')


def audit_attempt(run,parsed,submission):
    attempt=parsed.get('attempt',{});manifest=submission['manifest']
    if not all(k in attempt for k in ['started_at','finished_at']):return dict(passed=False,reason='attempt_not_finished')
    start=timestamp(attempt['started_at']);end=timestamp(attempt['finished_at'])
    checks=[json.loads(s) for s in (run/'ownership_checks.jsonl').read_text().splitlines()]
    window=audit_window(checks,start,end,manifest['boot_id'],manifest['ownership_digest'])
    rows=[json.loads(s) for s in (run/'service/predictions.jsonl').read_text().splitlines()]
    handled=[r for r in rows if r.get('route_path')==manifest['prediction_path'] and
             start-1<=timestamp(r['received_at'])<=end+1 and r['sequence_id']!='local']
    sequences={r['sequence_id'] for r in handled};indices={r['frame_index'] for r in handled}
    matching=bool(handled) and all(r.get('boot_id')==manifest['boot_id'] and r.get('ownership_digest')==manifest['ownership_digest'] and not r['error'] for r in handled)
    # A few verify requests cannot stand in for a complete approx250-frame run.
    sustained=len(handled)>=200 and len(sequences)==1 and max(indices,default=0)-min(indices,default=0)>=230
    result=dict(passed=bool(window['passed'] and matching and sustained),window=window,owned_frames=len(handled),
                one_owned_sequence=len(sequences)==1,frame_span=max(indices,default=0)-min(indices,default=0),
                expected_boot_id=manifest['boot_id'],route_path=manifest['prediction_path'])
    (run/'ownership_result_audit.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['probe','watch']);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=ROOT/args.run
    if args.command=='probe':print(json.dumps(probe(run,'manual')))
    else:
        while True:
            try:probe(run,'watch')
            except RuntimeError:pass
            time.sleep(2)


if __name__=='__main__':main()
