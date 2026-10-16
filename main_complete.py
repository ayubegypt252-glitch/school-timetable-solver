from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ortools.sat.python import cp_model

app = FastAPI(title="School Timetable CP-SAT API", version="1.1.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins or ["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)


class SolveRequest(BaseModel):
    data: dict[str, Any]
    rules: dict[str, Any]
    time_limit_seconds: int = Field(default=30, ge=1, le=120)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_lab(text: str) -> bool:
    text = (text or "").lower()
    return any(word in text for word in ("lab", "science", "chemistry", "biology", "physics"))


def unavailable(teacher: dict, day: str, period: int) -> bool:
    slots = teacher.get("unavailable") or []
    return (
        day in (teacher.get("dayOffs") or [])
        or day in (teacher.get("unavailableDays") or [])
        or f"{day}|{period}" in slots
        or f"{day}_{period}" in slots
    )


def make_pattern(assignment: dict) -> list[tuple[int, int | None]]:
    n = as_int(assignment.get("periods"))
    if n == 3:
        return [(2, None), (1, 7)]
    if n == 4:
        return [(2, None), (2, None)]
    if n == 5:
        return [(2, None), (2, None), (1, 7)]
    if assignment.get("double"):
        parts = []
        while n >= 2:
            parts.append((2, None))
            n -= 2
        if n:
            parts.append((1, None))
        return parts
    return [(1, None)] * n


@app.get("/")
def root():
    return {"service": "School Timetable CP-SAT API", "status": "ok",
            "solver": "Google OR-Tools CP-SAT", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
def solve(req: SolveRequest):
    incoming = req.data
    rules = req.rules
    teachers = incoming.get("teachers") or []
    classes = incoming.get("classes") or []
    subjects = incoming.get("subjects") or []
    rooms = incoming.get("rooms") or []
    assignments = incoming.get("assignments") or []
    days = rules.get("days") or ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    periods = as_int(rules.get("periods"), 7)
    blocked = {as_int(x, -1) for x in (rules.get("blocked") or [])}

    if not isinstance(days, list) or not days:
        raise HTTPException(400, "Select at least one school day.")
    if periods < 1 or periods > 12:
        raise HTTPException(400, "Periods per day must be between 1 and 12.")
    if any(p < 1 or p > periods for p in blocked):
        raise HTTPException(400, "Blocked periods must be within the configured school day.")
    if any(as_int(a.get("periods")) in (3, 5) for a in assignments) and (periods < 7 or 7 in blocked):
        raise HTTPException(400, "Hard rule: 3- and 5-period assignments require an available Period 7.")

    tm = {x["id"]: x for x in teachers if x.get("id")}
    cm = {x["id"]: x for x in classes if x.get("id")}
    sm = {x["id"]: x for x in subjects if x.get("id")}
    errors = []
    valid = []
    ids = set()
    for a in assignments:
        aid = a.get("id")
        c, s, t = cm.get(a.get("classId")), sm.get(a.get("subjectId")), tm.get(a.get("teacherId"))
        n = as_int(a.get("periods"), -1)
        if not aid:
            errors.append("Every assignment must have an ID.")
        elif aid in ids:
            errors.append(f"Duplicate assignment ID: {aid}.")
        else:
            ids.add(aid)
        if not c or not s or not t:
            errors.append(f"Incomplete assignment {aid or '(no ID)'}: check class, subject and teacher links.")
            continue
        if n < 1:
            errors.append(f"{c.get('name')} / {s.get('name')}: periods must be a positive whole number.")
            continue
        qualified = [str(x).strip().casefold() for x in (t.get("subjects") or [])]
        if str(s.get("name", "")).strip().casefold() not in qualified:
            errors.append(f"{t.get('name', 'Teacher')} is not mapped to subject {s.get('name')}.")
            continue
        valid.append(a)

    required = sum(max(0, as_int(a.get("periods"))) for a in assignments)
    if errors or not valid:
        if not errors:
            errors = ["No valid lesson assignments were supplied."]
        return {"status": "INVALID_INPUT", "schedule": [], "missing": errors,
                "summary": {"scheduledPeriods": 0, "requiredPeriods": required,
                            "complete": False, "solver": "Google OR-Tools CP-SAT"}}

    model = cp_model.CpModel()
    blocks = []
    blocks_by_assignment = {}
    presences = {}

    for a in valid:
        aid = str(a["id"])
        c, s, t = cm[a["classId"]], sm[a["subjectId"]], tm[a["teacherId"]]
        present = model.NewBoolVar(f"present_{aid}")
        presences[aid] = present
        blocks_by_assignment[aid] = []
        for bi, (size, preferred) in enumerate(make_pattern(a)):
            key = f"{aid}_{bi}"
            block = {"aid": aid, "c": c, "s": s, "t": t, "size": size,
                     "preferred": preferred, "key": key, "choices": []}
            groups = []
            for day in days:
                for start in range(1, periods + 1):
                    if preferred is not None and start != preferred:
                        continue
                    group = [(day, start + offset) for offset in range(size)]
                    if any(p > periods or p in blocked for _, p in group):
                        continue
                    if any(unavailable(t, d, p) for d, p in group):
                        continue
                    groups.append(group)
            if not groups:
                desc = "consecutive double" if size == 2 else ("single in Period 7" if preferred == 7 else "single period")
                errors.append(f"{c.get('name')} / {s.get('name')}: no legal placement for {desc}.")
            for ci, group in enumerate(groups):
                x = model.NewBoolVar(f"x_{aid}_{bi}_{ci}")
                block["choices"].append((x, group))
            model.Add(sum(x for x, _ in block["choices"]) == present)
            blocks.append(block)
            blocks_by_assignment[aid].append(block)

    if errors:
        return {"status": "INVALID_INPUT", "schedule": [], "missing": errors,
                "summary": {"scheduledPeriods": 0, "requiredPeriods": required,
                            "complete": False, "solver": "Google OR-Tools CP-SAT"}}

    # Class and teacher may each have at most one lesson in a time slot.
    occupancy = {}
    for b in blocks:
        for x, group in b["choices"]:
            for day, period in group:
                occupancy.setdefault(("c", b["c"]["id"], day, period), []).append(x)
                occupancy.setdefault(("t", b["t"]["id"], day, period), []).append(x)
    for variables in occupancy.values():
        if len(variables) > 1:
            model.AddAtMostOne(variables)

    # Workload caps count occupied periods, including both periods in a double.
    for t in teachers:
        tid = t.get("id")
        if not tid:
            continue
        weekly = []
        for b in blocks:
            if b["t"]["id"] == tid:
                for x, group in b["choices"]:
                    weekly.extend([x] * len(group))
        max_week = as_int(t.get("maxWeek"), 24)
        if weekly and max_week > 0:
            model.Add(sum(weekly) <= max_week)
        max_day = as_int(t.get("maxDay"), 5)
        if max_day > 0:
            for day in days:
                daily = []
                for b in blocks:
                    if b["t"]["id"] == tid:
                        for x, group in b["choices"]:
                            daily.extend([x] * sum(1 for d, _ in group if d == day))
                if daily:
                    model.Add(sum(daily) <= max_day)

    # Room assignment is optional, but any assigned room must be suitable and unique per slot.
    room_vars = {}
    room_occ = {}
    for b in blocks:
        eligible = []
        for r in rooms:
            cap, students = as_int(r.get("capacity")), as_int(b["c"].get("students"))
            if cap > 0 and students > 0 and cap < students:
                continue
            if is_lab(b["s"].get("name", "")) and not is_lab(f"{r.get('type', '')} {r.get('name', '')}"):
                continue
            eligible.append(r)
        preferred_room = next((r for r in eligible if r.get("name") == b["c"].get("room")), None)
        if preferred_room:
            eligible = [preferred_room] + [r for r in eligible if r.get("id") != preferred_room.get("id")]
        for x, group in b["choices"]:
            if not eligible:
                continue
            room_choices = []
            for ri, room in enumerate(eligible):
                z = model.NewBoolVar(f"room_{b['key']}_{ri}")
                room_choices.append(z)
                room_vars[(id(x), str(room["id"]))] = z
                for day, period in group:
                    room_occ.setdefault((room["id"], day, period), []).append(z)
            model.Add(sum(room_choices) == x)
    for variables in room_occ.values():
        if len(variables) > 1:
            model.AddAtMostOne(variables)

    model.Maximize(sum(as_int(a.get("periods")) * presences[str(a["id"])] for a in valid))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(5, min(120, req.time_limit_seconds))
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        missing = [
            f"{cm[a['classId']]['name']} / {sm[a['subjectId']]['name']} "
            f"({tm[a['teacherId']]['name']}): complete required pattern could not fit."
            for a in valid
        ]
        return {"status": status_name, "schedule": [], "missing": missing,
                "summary": {"scheduledPeriods": 0, "requiredPeriods": required,
                            "complete": False, "solver": "Google OR-Tools CP-SAT",
                            "wallTimeSeconds": round(solver.WallTime(), 2)}}

    schedule = []
    missing = []
    for a in valid:
        aid = str(a["id"])
        if solver.Value(presences[aid]) == 0:
            missing.append(f"{cm[a['classId']]['name']} / {sm[a['subjectId']]['name']} "
                           f"({tm[a['teacherId']]['name']}): complete pattern could not be placed.")
            continue
        for b in blocks_by_assignment[aid]:
            for x, group in b["choices"]:
                if not solver.Value(x):
                    continue
                chosen_room = ""
                for room in rooms:
                    z = room_vars.get((id(x), str(room.get("id"))))
                    if z is not None and solver.Value(z):
                        chosen_room = room["id"]
                        break
                for day, period in group:
                    schedule.append({"id": f"{aid}_{day}_{period}",
                                     "assignmentId": aid,
                                     "pairingBlockId": b["key"],
                                     "pairingBlockSize": b["size"],
                                     "classId": b["c"]["id"],
                                     "subjectId": b["s"]["id"],
                                     "teacherId": b["t"]["id"],
                                     "roomId": chosen_room,
                                     "day": day, "period": period})

    # Independent final checks: do not publish if any hard constraint fails.
    final_errors = []
    by_assignment = {}
    seen_class, seen_teacher, seen_room = set(), set(), set()
    for lesson in schedule:
        by_assignment.setdefault(str(lesson["assignmentId"]), []).append(lesson)
        day, p = str(lesson["day"]), as_int(lesson["period"])
        ck = (str(lesson["classId"]), day, p)
        tk = (str(lesson["teacherId"]), day, p)
        rk = (str(lesson.get("roomId") or ""), day, p)
        if ck in seen_class:
            final_errors.append(f"Class double-booking on {day}, period {p}.")
        if tk in seen_teacher:
            final_errors.append(f"Teacher double-booking on {day}, period {p}.")
        if lesson.get("roomId") and rk in seen_room:
            final_errors.append(f"Room double-booking on {day}, period {p}.")
        seen_class.add(ck)
        seen_teacher.add(tk)
        if lesson.get("roomId"):
            seen_room.add(rk)
        if unavailable(tm[lesson["teacherId"]], day, p):
            final_errors.append(f"{tm[lesson['teacherId']]['name']} is scheduled while unavailable on {day}, period {p}.")
        if p in blocked:
            final_errors.append(f"A lesson was placed in blocked period {p}.")

    for a in valid:
        aid, n = str(a["id"]), as_int(a.get("periods"))
        c_name, s_name = cm[a["classId"]]["name"], sm[a["subjectId"]]["name"]
        lessons = by_assignment.get(aid, [])
        if len(lessons) != n:
            final_errors.append(f"{c_name} / {s_name}: expected {n} periods, found {len(lessons)}.")
            continue
        if n not in (3, 4, 5):
            continue
        groups = {}
        for lesson in lessons:
            groups.setdefault(str(lesson.get("pairingBlockId")), []).append(lesson)
        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        expected = {3: [2, 1], 4: [2, 2], 5: [2, 2, 1]}[n]
        if sizes != expected:
            final_errors.append(f"{c_name} / {s_name}: invalid pairing {sizes}; expected {expected}.")
            continue
        for group in groups.values():
            if len(group) == 2 and (group[0]["day"] != group[1]["day"] or
                                    abs(as_int(group[0]["period"]) - as_int(group[1]["period"])) != 1):
                final_errors.append(f"{c_name} / {s_name}: double periods are not consecutive on one day.")
        if n in (3, 5):
            p7 = [e for e in lessons if as_int(e["period"]) == 7]
            singles = [g for g in groups.values() if len(g) == 1]
            if len(p7) != 1 or len(singles) != 1 or as_int(singles[0][0]["period"]) != 7:
                final_errors.append(f"{c_name} / {s_name}: exactly one single lesson must be in Period 7.")

    placed = len(schedule)
    complete = placed == required and not missing and not final_errors
    if final_errors:
        return {"status": "VALIDATION_FAILED", "schedule": [], "missing": final_errors + missing,
                "summary": {"scheduledPeriods": placed, "requiredPeriods": required,
                            "complete": False, "solver": "Google OR-Tools CP-SAT",
                            "message": "Final validation failed; timetable was not published.",
                            "wallTimeSeconds": round(solver.WallTime(), 2)}}

    return {"status": status_name, "schedule": schedule, "missing": missing,
            "summary": {"scheduledPeriods": placed, "requiredPeriods": required,
                        "complete": complete, "solver": "Google OR-Tools CP-SAT",
                        "wallTimeSeconds": round(solver.WallTime(), 2)}}
