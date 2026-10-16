"""Render every PDF with independent Poppler; make README and detail proofs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def resolve_poppler(requested: str | None) -> str:
    if requested:
        return requested
    system=shutil.which("pdftoppm")
    if system:
        return system
    bundled=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/bin/pdftoppm"
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("pdftoppm is unavailable; install Poppler or pass --pdftoppm")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pdftoppm")
    args=parser.parse_args();root=args.root.resolve()
    exe=resolve_poppler(args.pdftoppm)
    output=root/"checks/pdf-render";output.mkdir(exist_ok=True)
    layouts=json.loads((root/"layouts.json").read_text())
    ids=list(layouts["figures"])
    jobs=[]
    for profile in ("screen","paper"):
        folder=root/"figures" if profile=="screen" else root/"figures/paper"
        for fid in ids:
            jobs.append((profile,folder/(fid+".pdf"),output/(profile+"-"+fid)))

    def render(job: tuple) -> dict:
        profile,pdf,prefix=job
        command=[exe,"-png","-singlefile","-r","150",str(pdf),str(prefix)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=30)
        return {"profile":profile,"file":str(pdf.relative_to(root)),
                "input_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "renderer":"Poppler pdftoppm -png -singlefile -r 150",
                "exit_code":run.returncode,"stderr":run.stderr.strip(),
                "rendered":str(prefix.relative_to(root))+".png"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(render,jobs))
    (root/"checks/pdf-render.json").write_text(json.dumps(results,indent=2)+"\n")
    if any(r["exit_code"] for r in results):
        raise RuntimeError("At least one independent PDF render failed; see checks/pdf-render.json")
    from PIL import Image
    proof=root/"checks/proofs";proof.mkdir(exist_ok=True)
    for fid in ids:
        with Image.open(root/"figures"/(fid+".png")) as im:
            im.load()
            h=round(im.height*1200/im.width)
            im.resize((1200,h),Image.Resampling.LANCZOS).save(proof/(fid+"-readme.png"))
            im.resize((1200,h),Image.Resampling.LANCZOS).convert("L").save(proof/(fid+"-grayscale.png"))
            # Model-independent geometry regions come from layouts, not a second architecture story.
            for key,region in layouts["figures"][fid].get("inspection_regions",{}).items():
                x,y,w,h=region
                crop=im.crop((2*x,2*y,2*(x+w),2*(y+h)))
                crop.thumbnail((1600,1200),Image.Resampling.LANCZOS)
                crop.save(proof/(fid+"-"+key+".png"))
    print(f"Independent PDF renders: {len(results)}/{len(jobs)}; README, grayscale and detail proofs written.")


if __name__=="__main__":
    main()
