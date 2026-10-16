"""Small editable SVG primitives; all architecture text comes from the model."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def element(tag: str, parent: ET.Element, **attrs: Any) -> ET.Element:
    normalized = {k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None}
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", normalized)


class Canvas:
    def __init__(self, figure: dict, layout: dict, style: dict, profile: str) -> None:
        import cairocffi as cairo
        self.cairo = cairo
        self.figure, self.layout, self.style = figure, layout, style
        self.profile = style["profiles"][profile]
        self.width, self.height = layout["canvas"]
        self.root = ET.Element(f"{{{SVG}}}svg", {
            "id": figure["id"], "width": str(self.width), "height": str(self.height),
            "viewBox": f"0 0 {self.width} {self.height}", "role": "img",
            "aria-labelledby": figure["id"] + "_accessible_title " + figure["id"] + "_accessible_desc",
            "font-family": style["font_family"],
        })
        element("title", self.root, id=figure["id"] + "_accessible_title").text = figure["title"]
        element("desc", self.root, id=figure["id"] + "_accessible_desc").text = figure["alt_en"]
        self.ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        self.geometry: dict[str, list] = {"nodes": [], "panels": [], "texts": [], "edges": []}
        self.defs = element("defs", self.root)
        element("rect", self.root, id="background", x=0, y=0, width=self.width,
                height=self.height, fill="#FFFFFF")

    def data(self, obj: dict) -> dict[str, str]:
        return {"data_claim": " ".join(obj["claim_ids"]),
                "data_kind": obj["kind"], "data_status": obj["status"]}

    def metrics(self, text: str, size: float, bold: bool = False) -> tuple:
        c = self.cairo
        self.ctx.select_font_face(self.style["font_family"], c.FONT_SLANT_NORMAL,
                                  c.FONT_WEIGHT_BOLD if bold else c.FONT_WEIGHT_NORMAL)
        self.ctx.set_font_size(size)
        return self.ctx.text_extents(text)

    def text(self, parent: ET.Element, ident: str, lines: list[str], x: float, y: float,
             role: str = "secondary", anchor: str = "start", bold: bool = False,
             max_width: float | None = None, owner: str | None = None,
             color: str | None = None, claims: list[str] | None = None) -> None:
        size = self.profile["fonts"][role]
        line_gap = size * 1.25
        t = element("text", parent, id=ident, x=x, y=y, font_size=size,
                    text_anchor=anchor, fill=color or self.style["text"],
                    font_weight="bold" if bold else "normal",
                    data_claim=" ".join(claims or []), data_role=role)
        for i, line in enumerate(lines):
            yy = y + i * line_gap
            element("tspan", t, x=x, y=yy).text = line
            xb, yb, w, h, advance, _ = self.metrics(line, size, bold)
            origin = x - (advance / 2 if anchor == "middle" else advance if anchor == "end" else 0)
            box = [origin + xb, yy + yb, w, h]
            self.geometry["texts"].append({"id": f"{ident}:{i}", "text": line,
                "bbox": box, "font_px": size, "role": role, "owner": owner,
                "max_width": max_width, "advance": advance})

    def panel(self, obj: dict, placement: dict) -> None:
        x, y, w, h = placement["rect"]
        g = element("g", self.root, id=obj["id"], **self.data(obj))
        color, fill = self.style["colors"][obj["kind"]]
        element("rect", g, id=obj["id"] + "_boundary", x=x, y=y, width=w,
                height=h, rx=14, fill=fill if obj.get("filled") else "#FFFFFF",
                stroke=color, stroke_width=2, stroke_dasharray=self.style["dash"][obj["status"]])
        self.text(g, obj["id"] + "_label", obj["label"], x+24, y+48, "panel",
                  bold=True, max_width=w-48, owner=obj["id"], claims=obj["claim_ids"])
        if obj.get("subtitle"):
            self.text(g, obj["id"] + "_subtitle", obj["subtitle"], x+24, y+94,
                      max_width=w-48, owner=obj["id"], claims=obj["claim_ids"])
        self.geometry["panels"].append({"id": obj["id"], "bbox": [x,y,w,h]})

    def node(self, obj: dict, placement: dict, status_labels: dict) -> None:
        x, y, w, h = placement["rect"]
        g = element("g", self.root, id=obj["id"], **self.data(obj))
        color, fill = self.style["colors"][obj["kind"]]
        element("rect", g, id=obj["id"] + "_box", x=x, y=y, width=w, height=h,
                rx=12, fill=fill, stroke=color, stroke_width=2.5,
                stroke_dasharray=self.style["dash"][obj["status"]])
        if obj.get("badge", True):
            badge = status_labels[obj["status"]]
            bsize = self.profile["fonts"]["badge"]
            bw = self.metrics(badge, bsize, True)[4] + 22
            element("rect", g, id=obj["id"] + "_badge_box", x=x+12, y=y+10,
                    width=bw, height=bsize+10, rx=5, fill="#FFFFFF")
            self.text(g, obj["id"] + "_badge", [badge], x+23, y+11+bsize, "badge",
                      bold=True, owner=obj["id"], max_width=w-24, color=color,
                      claims=obj["claim_ids"])
            extra = bsize+18
        else:
            extra = 0
        size = self.profile["fonts"]["label"]
        lines = obj["label"]
        baseline = y + extra + (h-extra)/2 + size*.34 - (len(lines)-1)*size*.625
        self.text(g, obj["id"] + "_text", lines, x+w/2, baseline, "label", "middle",
                  obj.get("bold", False), w-30, obj["id"], claims=obj["claim_ids"])
        self.geometry["nodes"].append({"id":obj["id"], "bbox":[x,y,w,h],
                                       "status":obj["status"], "kind":obj["kind"]})

    def edge(self, obj: dict, placement: dict) -> None:
        color = self.style["colors"][obj["kind"]][0]
        marker_id = "arrow_" + obj["kind"]
        if self.defs.find(f"{{{SVG}}}marker[@id='{marker_id}']") is None:
            m = element("marker", self.defs, id=marker_id, markerWidth=10,
                        markerHeight=10, refX=9, refY=5, orient="auto", markerUnits="userSpaceOnUse")
            element("path", m, d="M 0 0 L 10 5 L 0 10 Z", fill=color)
        points = placement["points"]
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in points)
        g = element("g", self.root, id=obj["id"], **self.data(obj))
        element("path", g, id=obj["id"] + "_path", d=d, fill="none", stroke=color,
                stroke_width=3.5 if obj["kind"]=="feedback" else 2.5,
                stroke_dasharray=self.style["dash"][obj["status"]],
                marker_end="url(#"+marker_id+")", stroke_linejoin="round")
        if obj.get("label"):
            p = placement["label"]
            self.text(g, obj["id"] + "_label", obj["label"], p[0], p[1], "edge",
                      placement.get("anchor", "middle"), max_width=placement.get("label_width"),
                      owner=obj["id"], color=color, claims=obj["claim_ids"])
        self.geometry["edges"].append({"id":obj["id"], "points":points,
                                       "source":obj["source"], "target":obj["target"]})

    def annotation(self, obj: dict, placement: dict) -> None:
        g = element("g", self.root, id=obj["id"], **self.data(obj))
        if "rect" in placement:
            x,y,w,h = placement["rect"]
            color,fill = self.style["colors"][obj["kind"]]
            element("rect",g,id=obj["id"]+"_box",x=x,y=y,width=w,height=h,rx=10,
                    fill=fill,stroke=color,stroke_width=2,
                    stroke_dasharray=self.style["dash"][obj["status"]])
            self.geometry["panels"].append({"id":obj["id"],"bbox":[x,y,w,h]})
        self.text(g,obj["id"]+"_text",obj["label"],*placement["xy"],
                  obj.get("role","secondary"), placement.get("anchor","start"),
                  obj.get("bold",False),placement.get("max_width"),
                  obj["id"],claims=obj["claim_ids"])

    def legend(self, status_labels: dict, status_names: dict, semantic_legend: list) -> None:
        # Object colors and implementation states are two separate legends.
        y = self.height - 132
        widths = (self.width-96)/len(semantic_legend)
        for index, item in enumerate(semantic_legend):
            x=48+index*widths
            color, fill=self.style["colors"][item["kind"]]
            g=element("g",self.root,id="color_legend_"+item["kind"],data_kind="legend")
            element("rect",g,id="color_sample_"+item["kind"],x=x,y=y-22,width=30,height=30,
                    rx=5,fill=fill,stroke=color,stroke_width=2.5)
            self.text(g,"color_label_"+item["kind"],[item["label"]],x+44,y+3,
                      "legend",max_width=widths-48)
        y = self.height - 72
        widths = (self.width-96)/4
        for index, status in enumerate(("implemented","documented","proposed","unknown")):
            x = 48 + index*widths
            g = element("g",self.root,id="legend_"+status,data_status=status,data_kind="legend")
            element("rect",g,id="legend_"+status+"_sample",x=x,y=y-24,width=48,height=32,
                    rx=5,fill="#FFFFFF",stroke="#667085",stroke_width=2.5,
                    stroke_dasharray=self.style["dash"][status])
            self.text(g,"legend_"+status+"_text",[status_names[status]],x+64,y+3,
                      "legend",bold=True,max_width=widths-70)

    def bytes(self) -> bytes:
        ET.indent(self.root, space="  ")
        return ET.tostring(self.root, encoding="utf-8", xml_declaration=True)+b"\n"
