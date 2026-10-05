#!/usr/bin/env python3
"""Sequential unbiased diagnostics; never promotes a failed thermal pilot."""
from __future__ import annotations
import argparse
import re
import numpy as np
import run_reactive_solvent_contrast as pilot

engine = pilot.engine
OUT = pilot.BASE / 'reactive-1a-solvent-validation-v2'
PROTOCOL = dict(step_fs=1., hmass=4, shake=1, temperature_K=298.15,
                segment_time_ps=.5, maximum_segments=8, minimum_segments=2,
                consecutive_acceptable_segments=2, temperature_tolerance_fraction=.02,
                threads=4, sccacc=1., dump_fs=20., sdump_fs=100.)


def read_restart(path):
    rows = path.read_text().splitlines()[1:]
    arr = np.array([[float(t.replace('D', 'E')) for t in row.split()] for row in rows])
    if arr.shape != (241, 6) or not np.isfinite(arr).all():
        raise ValueError('Malformed native restart')
    return arr


def xyz_text(coords, label):
    elements = [a['element'] for a in pilot.STATE_DATA['atoms']]
    return '241\n' + label + '\n' + ''.join(
        f'{e:3s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n' for e,p in zip(elements, coords))


def md_input():
    return pilot.constraints() + ('$md\n  time=.5\n  step=1.0\n  dump=20.0\n'
        '  sdump=100.0\n  temp=298.15\n  nvt=true\n  restart=true\n'
        '  hmass=4\n  shake=1\n  sccacc=1.0\n$end\n')


def temperature(log):
    found = re.search(r'^\s*T\s*:\s*([\d.Ee+-]+)', log, re.M)
    if not found:
        raise ValueError('Native average temperature missing')
    return float(found.group(1))


def audit_structure(path):
    from dmap.models import Ensemble
    from dmap.chemistry import screen_topology
    from demonstrate_reactive_solvent_contrast import chiral_integrity
    raw=engine.read_xyz_ensemble(path)
    source=engine.read_xyz_ensemble(pilot.SOURCE).geometries[0]
    findings={}
    for center in (0,119):
        checked=screen_topology(Ensemble((source,)+raw.geometries),center_index=center)
        findings[str(center+1)]=len(checked.findings)
        if checked.findings:
            raise RuntimeError(f'Topology findings at Ru {center+1}: structural gate failed')
    chirality=chiral_integrity(np.array([g.coordinates for g in raw.geometries]))
    return dict(topology_findings_at_ru_centres=findings, donor_handedness=chirality,
                native_saved_frame_count=raw.size, trajectory_sha256=engine.digest(path))


def diagnose(state, manifest):
    info = manifest['states'].setdefault(state, {'segments': [], 'status': 'pending'})
    if info['status'] == 'passed':
        return
    initial = pilot.OUT / state / 'solvent-md.mdrestart'
    prior = initial
    info.setdefault('started_at', engine.now())
    info['status'] = 'running'
    for i in range(1, PROTOCOL['maximum_segments']+1):
        folder = OUT / 'diagnostics' / state / f'segment-{i:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        arr = read_restart(prior)
        engine.immutable(folder / 'input.xyz', xyz_text(arr[:,:3]*.529177210903,
            'native checkpoint geometry; unbiased temperature diagnostic'))
        engine.immutable(folder / 'md.inp', md_input())
        engine.immutable(folder / 'initial-mdrestart', prior.read_text())
        if not (folder / 'dynamics.complete.json').exists():
            engine.original.copy_input(folder / 'initial-mdrestart', folder / 'diagnostic.mdrestart')
        command = [str(engine.XTB), 'input.xyz', '--gfn', '2', '--chrg', '0', '--uhf', '0',
            '--cosmo', str(pilot.STATES[state]), '--parallel', '4', '--md',
            '--input', 'md.inp', '--namespace', 'diagnostic']
        engine.write(OUT / 'validation-manifest.json', manifest)
        log = engine.command(folder, 'dynamics', command)
        if 'normal exit of md()' not in log or 'RESTART' not in log:
            raise RuntimeError('Incomplete native diagnostic')
        if 'all: F' not in log or 'not recognized' in log.lower():
            raise RuntimeError('Hydrogen-only SHAKE/input not confirmed')
        t = temperature(log)
        traj = engine.original.validate_trajectory(folder / 'diagnostic.xtb.trj',
            [a['element'] for a in pilot.STATE_DATA['atoms']], pilot.PAIRS)
        acceptable = abs(t/298.15-1) <= .02 and 'thermostating problem' not in log
        row = dict(segment=i, average_temperature_K=t, temperature_gate=acceptable,
            input_restart_sha256=engine.digest(prior),
            output_restart_sha256=engine.digest(folder / 'diagnostic.mdrestart'),
            trajectory_sha256=engine.digest(folder / 'diagnostic.xtb.trj'),
            trajectory_validation=traj,
            native_degrees_of_freedom=int(re.search(r'# deg\. of freedom\s*:\s*(\d+)',log).group(1)))
        if len(info['segments']) < i:
            info['segments'].append(row)
        elif info['segments'][i-1] != row:
            raise RuntimeError('Completed diagnostic changed')
        engine.write(OUT / 'validation-manifest.json', manifest)
        print(f'{state} segment {i}: mean T={t:.3f} K; gate={acceptable}', flush=True)
        prior = folder / 'diagnostic.mdrestart'
        if i >= 2 and all(r['temperature_gate'] for r in info['segments'][-2:]):
            info.update(status='passed', completed_at=engine.now(),
                interpretation='Thermal diagnostic only; started from biased pilot endpoint; not an independent production replicate')
            engine.write(OUT / 'validation-manifest.json', manifest)
            return
    info.update(status='failed_temperature_gate', completed_at=engine.now())
    engine.write(OUT / 'validation-manifest.json', manifest)
    raise RuntimeError('Temperature gate not met; do not start independent production')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', choices=list(pilot.STATES))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / 'validation-manifest.json'
    manifest = engine.read(path) if path.exists() else dict(schema='case3d.reactive-validation.v2',
        created_at=engine.now(), status='diagnosing', protocol=PROTOCOL, states={},
        source_sha256=engine.digest(pilot.SOURCE),
        scope='Same 1a-derived Ru-allenylidene/4d/OTs assembly; matched dielectric surrogates; no DFT/TS/selectivity fitting')
    if manifest['protocol'] != PROTOCOL:
        raise RuntimeError('Changed versioned protocol')
    try:
        if args.audit_only:
            audits={}
            for state,info in manifest['states'].items():
                for row in info['segments']:
                    key=f"{state}/segment-{row['segment']:02d}"
                    audits[key]=audit_structure(OUT/'diagnostics'/key/'diagnostic.xtb.trj')
            engine.write(OUT/'structural-audit.json',dict(status='passed_for_completed_segments',audits=audits))
            if len(manifest['states'])==2 and all(s['status']=='passed' for s in manifest['states'].values()):
                manifest['status']='thermal_and_structural_diagnostics_passed'
            return
        if args.state and any(state!=args.state and info['status']=='running'
                              for state,info in manifest['states'].items()):
            raise RuntimeError('Another dielectric diagnostic is running; sequential execution required')
        for state in ([args.state] if args.state else pilot.STATES):
            diagnose(state, manifest)
        if len(manifest['states']) == 2 and all(s['status']=='passed' for s in manifest['states'].values()):
            manifest['status']='thermal_diagnostics_passed_pending_topology_review'
    except (Exception, KeyboardInterrupt) as err:
        manifest.update(status='interrupted' if isinstance(err,KeyboardInterrupt) else 'diagnostic_failed', error=str(err))
        raise
    finally:
        manifest['updated_at']=engine.now()
        engine.write(path,manifest)


if __name__ == '__main__':
    main()
