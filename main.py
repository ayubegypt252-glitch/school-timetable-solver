from __future__ import annotations
import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ortools.sat.python import cp_model

app = FastAPI(title='School Timetable CP-SAT API', version='1.0.0')
_origins = [x.strip() for x in os.getenv('ALLOWED_ORIGINS', '*').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

class SolveRequest(BaseModel):
    data: dict[str, Any]
    rules: dict[str, Any]
    time_limit_seconds: int = 30

@app.get('/')
def root():
    return {'service':'School Timetable CP-SAT API','status':'ok','solver':'Google OR-Tools CP-SAT'}

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/solve')
def solve(req: SolveRequest):
    data, rules = req.data, req.rules
    teachers = data.get('teachers', []); classes = data.get('classes', []); subjects = data.get('subjects', [])
    rooms = data.get('rooms', []); assignments = data.get('assignments', [])
    days = rules.get('days') or ['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']
    periods = int(rules.get('periods', 7)); blocked = set(int(x) for x in rules.get('blocked', []))
    if not days or periods < 1: raise HTTPException(400, 'Select at least one school day and one period.')
    if any(int(a.get('periods', 0)) in (3,5) for a in assignments) and (periods < 7 or 7 in blocked):
        raise HTTPException(400, 'Hard rule: assignments with 3 or 5 periods require an available Period 7.')
    tm={x['id']:x for x in teachers}; cm={x['id']:x for x in classes}; sm={x['id']:x for x in subjects}; rm={x['id']:x for x in rooms}
    slots=[(d,p) for d in days for p in range(1,periods+1) if p not in blocked]
    model=cp_model.CpModel(); blocks=[]; assignment_blocks={}; errors=[]
    # One assignment-presence variable means an assignment is scheduled completely or not at all.
    # This preserves required pairing structures even when the overall model is infeasible.
    for a in assignments:
        aid=a.get('id'); c=cm.get(a.get('classId')); s=sm.get(a.get('subjectId')); t=tm.get(a.get('teacherId')); n=int(a.get('periods') or 0)
        if not aid or not c or not s or not t:
            errors.append(f"Incomplete assignment: {aid or '(no ID)'}; check class, subject and teacher links."); continue
        if s['name'] not in (t.get('subjects') or []):
            errors.append(f"{t.get('name','Teacher')} is not mapped to subject {s['name']}."); continue
        if n < 1: errors.append(f"{c['name']} / {s['name']}: weekly periods must be at least 1."); continue
        if n==3: parts=[(2,None),(1,7)]
        elif n==4: parts=[(2,None),(2,None)]
        elif n==5: parts=[(2,None),(2,None),(1,7)]
        elif a.get('double'):
            parts=[]; rem=n
            while rem>=2: parts.append((2,None)); rem-=2
            if rem: parts.append((1,None))
        else: parts=[(1,None)]*n
        pres=model.NewBoolVar(f"present_{aid}"); assignment_blocks[aid]=[]
        for bi,(size,preferred) in enumerate(parts):
            b={'aid':aid,'a':a,'c':c,'s':s,'t':t,'size':size,'preferred':preferred,'pres':pres,'choices':[],'key':f'{aid}_{bi}'}
            legal_starts=[]
            for d in days:
                for p in range(1,periods+1):
                    group=[(d,p+i) for i in range(size)]
                    if preferred is not None and p != preferred: continue
                    if any(pp not in range(1,periods+1) or pp in blocked for _,pp in group): continue
                    if any(d in (t.get('dayOffs') or []) for _ in group): continue
                    if any(f"{dd}|{pp}" in (t.get('unavailable') or []) or f"{dd}_{pp}" in (t.get('unavailable') or []) for dd,pp in group): continue
                    legal_starts.append(group)
            if not legal_starts:
                errors.append(f"{c['name']} / {s['name']}: no legal placement exists for {'a consecutive double' if size==2 else ('Period 7 single' if preferred==7 else 'a single period')}.")
            for ci,group in enumerate(legal_starts):
                x=model.NewBoolVar(f"x_{aid}_{bi}_{ci}"); b['choices'].append((x,group))
            # Sum choices equals presence; if assignment is omitted, every block is omitted together.
            model.Add(sum(x for x,_ in b['choices']) == pres)
            blocks.append(b); assignment_blocks[aid].append(b)
    if errors:
        return {'status':'INVALID_INPUT','schedule':[],'missing':errors,'summary':{'scheduledPeriods':0,'requiredPeriods':sum(int(a.get('periods') or 0) for a in assignments),'solver':'OR-Tools CP-SAT'}}
    # No simultaneous class or teacher double-booking.
    occ={}
    for b in blocks:
        for x,group in b['choices']:
            for d,p in group:
                occ.setdefault(('class',b['c']['id'],d,p),[]).append(x)
                occ.setdefault(('teacher',b['t']['id'],d,p),[]).append(x)
    for vs in occ.values():
        if len(vs)>1: model.AddAtMostOne(vs)
    # Teacher workload caps and room occupancy (rooms assigned below based on class home-room preference).
    for t in teachers:
        tv=[x for b in blocks if b['t']['id']==t['id'] for x,g in b['choices'] for d,p in g]
        if tv and int(t.get('maxWeek') or 24)>0: model.Add(sum(tv) <= int(t.get('maxWeek') or 24))
        for d in days:
            dv=[x for b in blocks if b['t']['id']==t['id'] for x,g in b['choices'] for dd,p in g if dd==d]
            if dv and int(t.get('maxDay') or 5)>0: model.Add(sum(dv) <= int(t.get('maxDay') or 5))
    # Rooms are optional; enforce capacity and no double-booking when a suitable room is available.
    room_vars={}; room_occ={}
    for b in blocks:
        c=b['c']; s=b['s']; eligible=[]
        for r in rooms:
            if r.get('capacity') and c.get('students') and int(r['capacity']) < int(c['students']): continue
            if re_lab(s['name']) and not re_lab((r.get('type','')+' '+r.get('name',''))): continue
            eligible.append(r)
        preferred=next((r for r in eligible if r['name']==c.get('room')), None)
        if preferred: eligible=[preferred]+[r for r in eligible if r['id']!=preferred['id']]
        for x,group in b['choices']:
            if not eligible: continue
            rv=[]
            for ri,r in enumerate(eligible):
                z=model.NewBoolVar(f"room_{b['key']}_{id(x)}_{ri}"); rv.append(z)
                for d,p in group: room_occ.setdefault((r['id'],d,p),[]).append(z)
                room_vars[(x,r['id'])]=z
            model.Add(sum(rv)==x)
    for vs in room_occ.values():
        if len(vs)>1: model.AddAtMostOne(vs)
    # Maximize fully scheduled assignments first; then gently prefer even daily spread and earlier room preferences.
    scheduled_periods=sum(int(a.get('periods') or 0)*assignment_blocks[a['id']][0]['pres'] for a in assignments if a.get('id') in assignment_blocks and assignment_blocks[a['id']])
    model.Maximize(scheduled_periods)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=max(5,min(120,int(req.time_limit_seconds or 30))); solver.parameters.num_search_workers=8
    status=solver.Solve(model)
    status_name=solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        missing=[f"{cm[a['classId']]['name']} / {sm[a['subjectId']]['name']} ({tm[a['teacherId']]['name']}): could not fit the complete required pattern." for a in assignments if a.get('id') in assignment_blocks]
        return {'status':status_name,'schedule':[],'missing':missing,'summary':{'scheduledPeriods':0,'requiredPeriods':sum(int(a.get('periods') or 0) for a in assignments),'solver':'OR-Tools CP-SAT','message':'No feasible complete assignment set was found under the hard constraints.'}}
    schedule=[]; missing=[]
    for a in assignments:
        aid=a.get('id'); bs=assignment_blocks.get(aid,[])
        if not bs: continue
        if solver.Value(bs[0]['pres'])==0:
            missing.append(f"{cm[a['classId']]['name']}: {sm[a['subjectId']]['name']} ({tm[a['teacherId']]['name']}) could not be placed as a complete required pattern.")
            continue
        for b in bs:
            for x,group in b['choices']:
                if solver.Value(x):
                    chosen_room=''
                    for r in rooms:
                        z=room_vars.get((x,r['id']))
                        if z is not None and solver.Value(z): chosen_room=r['id']; break
                    for d,p in group:
                        schedule.append({'id':f"{aid}_{d}_{p}",'assignmentId':aid,'pairingBlockId':b['key'],'pairingBlockSize':b['size'],'classId':b['c']['id'],'subjectId':b['s']['id'],'teacherId':b['t']['id'],'roomId':chosen_room,'day':d,'period':p})
    required=sum(int(a.get('periods') or 0) for a in assignments); placed=len(schedule)
    return {'status':status_name,'schedule':schedule,'missing':missing,'summary':{'scheduledPeriods':placed,'requiredPeriods':required,'complete':placed==required and not missing,'solver':'Google OR-Tools CP-SAT','wallTimeSeconds':round(solver.WallTime(),2)}}

def re_lab(text: str) -> bool:
    return any(k in (text or '').lower() for k in ('lab','science','chemistry','biology','physics'))
