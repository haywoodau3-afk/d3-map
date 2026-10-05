from __future__ import annotations

import json
from pathlib import Path

from .io import read_xyz_ensemble


def write_setup_viewer(xyz: str | Path, output: str | Path) -> Path:
    """Write an offline, self-contained 3-D inspection companion for manual setup."""
    source, target = Path(xyz).resolve(), Path(output).resolve()
    geometry = read_xyz_ensemble(source).geometries[0]
    atoms = [
        {"number": index + 1, "element": element, "xyz": coordinate.tolist()}
        for index, (element, coordinate) in enumerate(zip(geometry.elements, geometry.coordinates))
    ]
    payload = json.dumps(atoms, separators=(",", ":"))
    target.write_text(
        f"""<!doctype html><html><head><meta charset=utf-8><title>d3map setup</title>
<style>body{{font:15px system-ui;margin:1rem;display:grid;grid-template-columns:620px 1fr;gap:1rem}}canvas{{background:#101827;border-radius:8px}}label{{display:block;margin:.5rem 0}}input{{width:95%}}.note{{color:#555}}</style></head>
<body><section><h1>d3map structure inspection</h1><canvas id=v width=600 height=600></canvas><p class=note>Drag to rotate. Hydrogens are displayed faintly and intentionally omitted from ordinary selectors.</p></section>
<section><h2>Atom roles</h2><label>Centre atom <select id=center></select></label><label>Frozen heavy atoms (comma-separated)<input id=frozen></label><label>Alignment heavy atoms<input id=align></label><label>Protected contacts, e.g. 1-4,1-9<input id=contacts></label><label>Reactive direction x,y,z<input id=direction value="0,0,1"></label><button id=save>Download reviewed setup</button><pre id=info></pre></section>
<script>const atoms={payload},heavy=atoms.filter(a=>a.element!=='H'),c=document.querySelector('#v'),x=c.getContext('2d');
for(const a of heavy)document.querySelector('#center').add(new Option(`${{a.number}} ${{a.element}}`,a.number));let ax=.3,ay=-.5,drag=false,last;
function draw(){{x.clearRect(0,0,600,600);const pts=atoms.map(a=>{{let [X,Y,Z]=a.xyz;let u=X*Math.cos(ay)+Z*Math.sin(ay),w=-X*Math.sin(ay)+Z*Math.cos(ay);let v=Y*Math.cos(ax)-w*Math.sin(ax),d=Y*Math.sin(ax)+w*Math.cos(ax);return{{a,u,v,d}}}}).sort((a,b)=>a.d-b.d);for(const p of pts){{let r=p.a.element==='H'?4:9;x.globalAlpha=p.a.element==='H'?.25:1;x.fillStyle=p.a.element==='Se'?'#ff9d2e':'#78b7ff';x.beginPath();x.arc(300+p.u*45,300-p.v*45,r,0,7);x.fill();x.fillStyle='white';x.fillText(p.a.number,310+p.u*45,296-p.v*45)}}x.globalAlpha=1}}draw();
c.onpointerdown=e=>{{drag=true;last=e}};c.onpointerup=()=>drag=false;c.onpointermove=e=>{{if(drag){{ay+=(e.clientX-last.clientX)/150;ax+=(e.clientY-last.clientY)/150;last=e;draw()}}}};
document.querySelector('#save').onclick=()=>{{const val=id=>document.querySelector(id).value;const cfg={{schema_version:'0.1',input:{json.dumps(source.name)},center_atom:+val('#center'),reactive_direction:val('#direction').split(',').map(Number),secondary_direction:[1,0,0],alignment_atoms:val('#align').split(',').filter(Boolean).map(Number),frozen_atoms:val('#frozen').split(',').filter(Boolean).map(Number),protected_contacts:val('#contacts').split(',').filter(Boolean).map(v=>v.split('-').map(Number)),topology:{{policy:'strict'}},population:{{model:'uniform'}},output:'dmap-results'}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(cfg,null,2)],{{type:'application/json'}}));a.download='project.yml';a.click()}};</script></body></html>""",
        encoding="utf-8",
    )
    return target
