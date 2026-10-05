"""Sequential, resumable xTB/CREST screening of the seven confirmed Case 4 complexes."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'02-dynamic-methodology'/'src'))
from dmap.io import read_xyz_ensemble
from dmap.sampling import XtbCrestConfig, executable_version, run_xtb_crest
ROOT=Path(__file__).resolve().parents[1]; FULL=ROOT/'structures/full-complexes'; OUT=ROOT/'sampling/runs'
XTB=ROOT.parents[1]/'.micromamba/part3c/bin/xtb'; CREST=ROOT.parents[1]/'.micromamba/part3c/bin/crest'
# Verified against atom-map CSV and the unrelaxed starting geometry.
AU, P, CU, CU_N, CARBENE = 2,1,73,(26,77,78),148
THREADS=4
units=json.loads((FULL/'manifest.json').read_text())['records']
all_units=units
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only',action='append',metavar='ID',help='run only this structure ID; may be repeated')
args=parser.parse_args()
if args.only:
 selected=set(args.only)
 known={entry['id'] for entry in all_units}
 unknown=selected-known
 if unknown:parser.error('unknown structure ID(s): '+', '.join(sorted(unknown)))
 units=[entry for entry in all_units if entry['id'] in selected]
(OUT).mkdir(parents=True,exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
manifest_path=OUT/'run-manifest.json'
manifest={'schema':'case4.xtb-crest.v2','status':'running','started_at':now(),'method':{'xtb':str(XTB),'xtb_version':executable_version(XTB),'crest':str(CREST),'crest_version':executable_version(CREST),'model':'GFN2-xTB','charge':2,'multiplicity':1,'solvent':'CHCl3 (ALPB)','preoptimization':'tight','crest_search':'quick','threads_per_complex':THREADS,'execution':'sequential,resumable'},'restraints':{'frozen_atoms':[AU],'protected_contacts':[[AU,CARBENE],[AU,P],*[ [CU,n] for n in CU_N]],'meaning':'Anchor the Au(I) position and protect Au-carbene, Au-phosphine, and three Cu-N coordination distances during xTB and CREST; Cu is not frozen, avoiding an artificial Au-Cu distance restraint.'},'structures':[]}
if manifest_path.exists():
 try:
  old=json.loads(manifest_path.read_text());
  if (old.get('schema')==manifest['schema'] and old.get('method')==manifest['method'] and old.get('restraints')==manifest['restraints']):
   manifest['structures']=old.get('structures',[])
 except (json.JSONDecodeError,OSError):pass
manifest['run_scope']=[entry['id'] for entry in units] if args.only else 'all'

def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
for entry in units:
 name=entry['id'];source=FULL/entry['file'];directory=OUT/name;stage=directory/'sampling';stage.mkdir(parents=True,exist_ok=True)
 previous=next((x for x in manifest['structures']if x['id']==name),None)
 if previous and previous.get('status')=='success'and Path(previous['ensemble']).is_file():continue
 # A prior user skip or a deliberately preserved partial checkpoint is not
 # silently restarted by a default whole-manifest run. Passing --only ID is
 # an explicit request to revisit that individual entry.
 if previous and previous.get('status') in {'skipped','checkpointed'} and not args.only:continue
 record={'id':name,'name':entry['name'],'input':str(source.relative_to(ROOT)),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_directory':str(directory.relative_to(ROOT)),'status':'running','started_at':now()}
 manifest['structures']=[x for x in manifest['structures']if x['id']!=name]+[record];save()
 try:
  result=run_xtb_crest(source,stage,config=XtbCrestConfig(charge=2,multiplicity=1,frozen_atom_numbers=(AU,),protected_contacts=((AU,CARBENE),(AU,P),*((CU,n)for n in CU_N)),solvent='chcl3',threads=THREADS,quick=True,allow_trajectory_fallback=True,metadyn_steps=25,metadyn_time_ps=5.0),xtb_executable=XTB,crest_executable=CREST)
  ens=read_xyz_ensemble(result.ensemble_xyz)
  assert len(ens.elements)==entry['atom_count'],(len(ens.elements),entry['atom_count'])
  record.update(status='success',completed_at=now(),preoptimized=str(result.preoptimized_xyz.relative_to(ROOT)),ensemble=str(result.ensemble_xyz.relative_to(ROOT)),conformers=ens.size,analysis_mode=result.analysis_mode,xtb_log=str(result.xtb_log.relative_to(ROOT)),crest_log=str(result.crest_log.relative_to(ROOT)),backend_versions=result.backend_versions,ensemble_sha256=hashlib.sha256(result.ensemble_xyz.read_bytes()).hexdigest())
 except Exception as exc:
  record.update(status='failed',completed_at=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();manifest.update(status='partial_failure',updated_at=now());save();raise
 save()
completed_ids={x['id'] for x in manifest['structures'] if x.get('status')=='success' and Path(x.get('ensemble','')).is_file()}
if {entry['id'] for entry in all_units}<=completed_ids:
 manifest.update(status='success',completed_at=now(),updated_at=now())
else:
 manifest.update(status='partial',updated_at=now());manifest.pop('completed_at',None)
save()
print(json.dumps({'status':manifest['status'],'structures':manifest['structures']},indent=2))
