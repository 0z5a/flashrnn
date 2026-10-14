"""Render JSON-driven, editable SVG and export screen and print artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from svg_primitives import Canvas


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing input JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_one(model: dict, layouts: dict, style: dict, figure: dict,
               profile: str) -> tuple[bytes, dict]:
    layout=layouts["figures"][figure["id"]]
    canvas=Canvas(figure,layout,style,profile)
    nodes={n["id"]:n for n in model["nodes"]}
    edges={e["id"]:e for e in model["edges"]}
    for obj in figure["panels"]:
        canvas.panel(obj,layout["panels"][obj["id"]])
    # Data edges sit below nodes; a route cannot hide inside an unrelated box.
    for ident in figure["edge_ids"]:
        canvas.edge(edges[ident],layout["edges"][ident])
    for ident in figure["node_ids"]:
        canvas.node(nodes[ident],layout["nodes"][ident],model["status_labels"])
    for obj in figure["annotations"]:
        canvas.annotation(obj,layout["annotations"][obj["id"]])
    canvas.legend(model["status_labels"],model["status_names"],model["semantic_legend"])
    return canvas.bytes(),canvas.geometry


def write_atomic(path: Path, body: bytes) -> None:
    """Readers see the previous complete artifact or the new complete artifact."""
    temporary=path.with_name(path.name+".partial")
    temporary.write_bytes(body)
    temporary.replace(path)


def export_pdf(svg: bytes, path: Path, width_mm: float) -> None:
    import cairosvg
    root=ET.fromstring(svg)
    _,_,w,h=map(float,root.attrib["viewBox"].split())
    root.set("width",f"{width_mm}mm")
    root.set("height",f"{width_mm*h/w}mm")
    write_atomic(path,cairosvg.svg2pdf(bytestring=ET.tostring(root)))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--layout",type=Path,required=True)
    p.add_argument("--style",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--formats",default="svg,png,pdf")
    args=p.parse_args()
    formats=set(args.formats.split(","))
    if not formats or not formats<= {"svg","png","pdf"}:
        p.error("--formats must contain only svg,png,pdf")
    model,layouts,style=(read_json(f) for f in (args.model,args.layout,args.style))
    if model.get("schema_version")!=1 or layouts.get("schema_version")!=1:
        raise ValueError("Unsupported JSON schema version")
    args.out.mkdir(parents=True,exist_ok=True)
    checks=args.model.resolve().parent/"checks"
    checks.mkdir(exist_ok=True)
    manifest={"formats":sorted(formats),"profiles":{},"inputs":{}}
    for f in (args.model,args.layout,args.style):
        manifest["inputs"][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    for profile in ("screen","paper"):
        dest=args.out if profile=="screen" else args.out/"paper"
        dest.mkdir(exist_ok=True)
        geometry={}
        outputs=[]
        for fig in model["figures"]:
            svg,geo=render_one(model,layouts,style,fig,profile)
            geometry[fig["id"]]=geo
            basename=fig["id"]
            if "svg" in formats:
                write_atomic(dest/(basename+".svg"),svg)
            if "png" in formats:
                import cairosvg
                write_atomic(dest/(basename+".png"),cairosvg.svg2png(bytestring=svg,scale=2))
            if "pdf" in formats:
                export_pdf(svg,dest/(basename+".pdf"),style["profiles"][profile]["pdf_width_mm"])
            for fmt in sorted(formats):
                path=dest/(basename+"."+fmt)
                outputs.append({"file":str(path.relative_to(args.out)),"bytes":path.stat().st_size,
                                "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"Rendered {profile}/{basename}: {','.join(sorted(formats))}")
        (checks/f"geometry-{profile}.json").write_text(json.dumps(geometry,indent=2)+"\n")
        manifest["profiles"][profile]={"outputs":outputs,"pdf_width_mm":style["profiles"][profile]["pdf_width_mm"]}
    (checks/"render-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
