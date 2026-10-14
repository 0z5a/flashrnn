"""Check evidence, editable SVG, geometry, logical DAGs and buffer lifetimes.

This validates only the diagram model. It does not qualify a CUDA kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from render_figures import read_json, render_one

STATES={"implemented","documented","proposed","unknown"}
EXPECTED=["fig01_architecture_overview","fig02_state_residency",
          "fig03_dependency_pipeline","fig04_execution_dispatch"]


def overlap(a: list, b: list, epsilon: float = .25) -> bool:
    return (min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]) > epsilon and
            min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]) > epsilon)


def contained(a: list, b: list, tolerance: float = 1.0) -> bool:
    return (a[0]>=b[0]-tolerance and a[1]>=b[1]-tolerance and
            a[0]+a[2]<=b[0]+b[2]+tolerance and a[1]+a[3]<=b[1]+b[3]+tolerance)


def segment_hits(a: list, b: list, box: list, inset: float = 1.5) -> bool:
    x,y,w,h=box
    x0,x1,y0,y1=x+inset,x+w-inset,y+inset,y+h-inset
    if abs(a[0]-b[0])<.01:
        return x0<a[0]<x1 and min(a[1],b[1])<y1 and max(a[1],b[1])>y0
    if abs(a[1]-b[1])<.01:
        return y0<a[1]<y1 and min(a[0],b[0])<x1 and max(a[0],b[0])>x0
    raise ValueError("Only orthogonal routes are supported by this validator")


def boundary(point: list, box: list) -> bool:
    x,y,w,h=box;px,py=point
    return ((abs(px-x)<1 or abs(px-x-w)<1) and y-1<=py<=y+h+1 or
            (abs(py-y)<1 or abs(py-y-h)<1) and x-1<=px<=x+w+1)


def schedule_errors(schedule: dict) -> list[str]:
    errors=[]
    events={e["id"]:e for e in schedule["events"]}
    if len(events)!=len(schedule["events"]):errors.append("Duplicate schedule event id")
    adj={e:[] for e in events};indegree={e:0 for e in events}
    for event in events.values():
        if event["start"]>event["end"]:errors.append(f"Negative event interval: {event['id']}")
    for dep in schedule["dependencies"]:
        a,b=dep["source"],dep["target"]
        if a not in events or b not in events:
            errors.append(f"Missing schedule event: {a}->{b}");continue
        if events[a]["end"]>events[b]["start"]:
            errors.append(f"Consumer before producer completion: {a}->{b}")
        adj[a].append(b);indegree[b]+=1
    queue=deque(sorted(k for k,v in indegree.items() if not v));seen=[]
    while queue:
        a=queue.popleft();seen.append(a)
        for b in adj[a]:
            indegree[b]-=1
            if not indegree[b]:queue.append(b)
    if len(seen)!=len(events):errors.append("Time-unrolled schedule is cyclic")
    # The static overview feedback graph intentionally is not checked for acyclicity.
    for buf in schedule["buffers"]:
        try:
            producer=events[buf["producer"]];ready=events[buf["ready"]]
            consumers=[events[x] for x in buf["consumers"]]
            reuse=events[buf["reuse"]] if buf.get("reuse") else None
        except KeyError as exc:
            errors.append(f"Missing buffer event: {buf['id']} / {exc}");continue
        if producer["end"]>ready["end"]:errors.append(f"Buffer ready before load: {buf['id']}")
        for consumer in consumers:
            if ready["end"]>consumer["start"]:errors.append(f"Buffer use before ready: {buf['id']}")
        if reuse and reuse["start"]<max(c["end"] for c in consumers):
            errors.append(f"Buffer overwrite before all consumers finish: {buf['id']}")
    return errors


def validate(root: Path) -> dict[str, Any]:
    model=read_json(root/"architecture.json");layouts=read_json(root/"layouts.json");style=read_json(root/"style.json")
    errors=[];warnings=[];counts=Counter();pdf_details=[]
    figures=model["figures"];claims={c["id"]:c for c in model["claims"]}
    if [f["id"] for f in figures]!=EXPECTED:errors.append("Four required figure basenames are not ordered uniquely")
    if len(claims)!=len(model["claims"]):errors.append("Duplicate claim id")
    source_records=model["repository_snapshot"]["sources"]
    # Source snapshots are portable. If the original tree is present, verify it too.
    for key,record in source_records.items():
        snap=root/record["snapshot"]
        if not snap.is_file() or hashlib.sha256(snap.read_bytes()).hexdigest()!=record["sha256"]:
            errors.append(f"Source snapshot changed: {key}")
        repo=root.parents[2]
        origin=repo/record["origin"]
        if origin.is_file() and hashlib.sha256(origin.read_bytes()).hexdigest()!=record["sha256"]:
            errors.append(f"Live source differs from captured evidence: {key}")
        counts["source_snapshots"]+=1
    for claim in claims.values():
        if claim["status"] not in STATES:errors.append(f"Invalid claim status: {claim['id']}")
        local=[e for e in claim["evidence"] if e["kind"]=="local_source"]
        if claim["status"]=="implemented" and not local:errors.append(f"Implemented claim lacks local code: {claim['id']}")
        if "profiled" in claim["verification"] and not any(e["kind"]=="raw_profile" for e in claim["evidence"]):
            errors.append(f"Profiled claim lacks raw trace: {claim['id']}")
        for ev in local:
            if ev.get("source_id") not in source_records:
                errors.append(f"Missing source record: {claim['id']}");continue
            record=source_records[ev["source_id"]]
            lo,hi=ev["lines"]
            if not 1<=lo<=hi<=record["line_count"]:errors.append(f"Invalid source line range: {claim['id']} {ev['source_id']}")
            if ev["sha256"]!=record["sha256"]:errors.append(f"Evidence SHA mismatch: {claim['id']}")
        for ev in claim["evidence"]:
            if ev["kind"] in ("compiler_output","local_measurement"):
                evidence_file=root/ev.get("snapshot",ev["locator"])
                if not evidence_file.is_file():
                    errors.append(f"Missing compiler/measurement file: {claim['id']}")
                elif hashlib.sha256(evidence_file.read_bytes()).hexdigest()!=ev["sha256"]:
                    errors.append(f"Compiler/measurement SHA mismatch: {claim['id']}")
            if ev["kind"]=="task_design":
                if not (root/ev["locator"].split(":section",1)[0]).is_file():
                    errors.append(f"Missing design file: {claim['id']}")
        counts["claims_"+claim["status"]]+=1
    node_map={n["id"]:n for n in model["nodes"]};edge_map={e["id"]:e for e in model["edges"]}
    if len(node_map)!=len(model["nodes"]) or len(edge_map)!=len(model["edges"]):errors.append("Duplicate semantic node/edge id")
    for fig in figures:
        if len(fig["node_ids"])!=len(set(fig["node_ids"])):errors.append(f"Duplicate node in {fig['id']}")
        objects=[node_map[i] for i in fig["node_ids"]]+[edge_map[i] for i in fig["edge_ids"]]+fig["panels"]+fig["annotations"]
        for obj in objects:
            if obj["status"] not in STATES:errors.append(f"Invalid object status: {obj['id']}")
            if not obj["claim_ids"] or not set(obj["claim_ids"])<=claims.keys():errors.append(f"Unknown claim on {obj['id']}")
        for ident in fig["edge_ids"]:
            e=edge_map[ident]
            if e["source"] not in fig["node_ids"] or e["target"] not in fig["node_ids"]:
                errors.append(f"Edge endpoint absent: {ident}")
        for lang in ["zh","en"]:
            headings=re.findall(r"(?m)^## (fig\d\d_\w+)$",(root/f"captions_{lang}.md").read_text())
            if headings!=EXPECTED:errors.append(f"Caption basename bijection failed: {lang}")
    errors.extend(schedule_errors(model["schedule"]))
    counts["schedule_events"]=len(model["schedule"]["events"])
    counts["schedule_dependencies"]=len(model["schedule"]["dependencies"])
    counts["buffer_versions"]=len(model["schedule"]["buffers"])
    for profile in ["screen","paper"]:
        folder=root/"figures" if profile=="screen" else root/"figures/paper"
        for fig in figures:
            fid=fig["id"];layout=layouts["figures"][fid];w,h=layout["canvas"]
            path=folder/(fid+".svg")
            if not path.is_file():errors.append(f"Missing SVG: {profile}/{fid}");continue
            svg=ET.fromstring(path.read_bytes())
            if "viewBox" not in svg.attrib or len(svg.attrib["viewBox"].split())!=4:errors.append(f"Invalid viewBox: {fid}")
            ids=[e.attrib["id"] for e in svg.iter() if "id" in e.attrib]
            if len(ids)!=len(set(ids)):errors.append(f"Duplicate SVG ids: {profile}/{fid}")
            if not svg.find("{*}title") is not None or svg.find("{*}desc") is None:errors.append(f"Missing title/desc: {fid}")
            for e in svg.iter():
                tag=e.tag.rsplit("}",1)[-1]
                if tag in ("script","foreignObject","image"):errors.append(f"Forbidden SVG tag: {tag} / {fid}")
                for k,v in e.attrib.items():
                    if k.endswith("href") and not v.startswith("#"):errors.append(f"External/embedded SVG resource: {fid}")
                    if "url(" in v and not re.fullmatch(r"url\(#[\w-]+\)",v):errors.append(f"External SVG url: {fid}")
            # The actual saved SVG must equal the model-generated version before using its measured geometry.
            rebuilt,geo=render_one(model,layouts,style,fig,profile)
            if path.read_bytes()!=rebuilt:errors.append(f"Saved SVG differs from model/style: {profile}/{fid}")
            for obj in [node_map[i] for i in fig["node_ids"]]+fig["panels"]+fig["annotations"]:
                if obj["status"] in ("proposed","unknown"):
                    group=svg.find(f".//{{*}}g[@id='{obj['id']}']")
                    label=" ".join(group.itertext()) if group is not None else ""
                    if model["status_labels"][obj["status"]] not in label:
                        errors.append(f"Proposal/TBD not visible: {obj['id']}")
            boxes={n["id"]:n["bbox"] for n in geo["nodes"]}
            for item in geo["nodes"]+geo["panels"]:
                if not contained(item["bbox"],[0,0,w,h]):errors.append(f"Object off canvas: {profile}/{item['id']}")
            nodes=geo["nodes"]
            for i,a in enumerate(nodes):
                for b in nodes[i+1:]:
                    if overlap(a["bbox"],b["bbox"]) and not contained(a["bbox"],b["bbox"]) and not contained(b["bbox"],a["bbox"]):
                        errors.append(f"Node overlap: {profile}/{a['id']} / {b['id']}")
            for t in geo["texts"]:
                if not contained(t["bbox"],[0,0,w,h]):errors.append(f"Text off canvas: {profile}/{t['id']}")
                if t["max_width"] is not None and t["advance"]>t["max_width"]+1:
                    errors.append(f"Text exceeds allocated width: {profile}/{t['id']} {t['advance']:.1f}>{t['max_width']}")
                if t["owner"] in boxes and not contained(t["bbox"],boxes[t["owner"]]):
                    errors.append(f"Text outside node: {profile}/{t['id']}")
                for other,b in boxes.items():
                    if other!=t["owner"] and overlap(t["bbox"],b):errors.append(f"Text covered by node: {profile}/{t['id']} / {other}")
                if profile=="paper":
                    pt=t["font_px"]*style["profiles"][profile]["pdf_width_mm"]/25.4*72/w
                    if pt<8:errors.append(f"Paper text below 8pt: {fid}/{t['id']}: {pt:.2f}pt")
            for i,a in enumerate(geo["texts"]):
                for b in geo["texts"][i+1:]:
                    if overlap(a["bbox"],b["bbox"]):errors.append(f"Text overlap: {profile}/{a['id']} / {b['id']}")
            for e in geo["edges"]:
                if not boundary(e["points"][0],boxes[e["source"]]) or not boundary(e["points"][-1],boxes[e["target"]]):
                    errors.append(f"Edge not anchored on boundary: {e['id']}")
                for a,b in zip(e["points"],e["points"][1:]):
                    if not contained([min(a[0],b[0]),min(a[1],b[1]),abs(a[0]-b[0]),abs(a[1]-b[1])],[0,0,w,h]):errors.append(f"Edge off canvas: {e['id']}")
                    for n,box in boxes.items():
                        if n not in (e["source"],e["target"]) and segment_hits(a,b,box):errors.append(f"Edge crosses non-target node: {profile}/{e['id']} / {n}")
                    for t in geo["texts"]:
                        if segment_hits(a,b,t["bbox"],0):errors.append(f"Edge obscures text: {profile}/{e['id']} / {t['id']}")
            from PIL import Image
            png=folder/(fid+".png")
            try:
                with Image.open(png) as im:
                    im.load()
                    if im.size!=(2*w,2*h):errors.append(f"Wrong 2x PNG dimensions: {profile}/{fid}")
            except Exception as exc:errors.append(f"PNG open failed: {profile}/{fid}: {exc}")
            from pypdf import PdfReader
            pdf=folder/(fid+".pdf")
            try:
                reader=PdfReader(pdf)
                if len(reader.pages)!=1:errors.append(f"PDF must have one page: {fid}")
                page=reader.pages[0]
                wanted=style["profiles"][profile]["pdf_width_mm"]/25.4*72
                if abs(float(page.mediabox.width)-wanted)>.1:errors.append(f"PDF physical width mismatch: {fid}")
                if len(page.images)!=0:errors.append(f"Raster image embedded in PDF: {fid}")
                text=page.extract_text()
                if len(text)<200:errors.append(f"PDF lacks real extractable text: {fid}")
                pdf_details.append({"profile":profile,"figure":fid,"page_width_pt":float(page.mediabox.width),
                    "page_height_pt":float(page.mediabox.height),"embedded_images":len(page.images),"extracted_text_chars":len(text),
                    "min_font_pt":min(t["font_px"] for t in geo["texts"])*wanted/w})
            except Exception as exc:errors.append(f"PDF open failed: {profile}/{fid}: {exc}")
            counts["svg_checked"]+=1;counts["text_lines_measured"]+=len(geo["texts"])
    return {"status":"PASS" if not errors else "FAIL","errors":errors,"warnings":warnings,
            "counts":dict(counts),"pdf_checks":pdf_details,
            "limitations":["Checks modeled dependencies only, not kernel correctness.",
                           "Source-level order is not a measured runtime timeline.",
                           "Visual inspection is recorded separately in review.md."]}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    report=validate(args.root.resolve())
    dest=args.root/"checks/validation.json";dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if report["status"]=="PASS" else 1)


if __name__=="__main__":
    main()
