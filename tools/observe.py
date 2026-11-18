"""Run current instrumentation over the intact approved portable model image."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import model

ROOT = Path(__file__).resolve().parents[1]


def verify_overlay(root, release):
    """Current scientific modules must remain byte-identical to approved files."""
    manifest=json.loads((release/'RELEASE.json').read_text(encoding='utf-8'))
    package=root/'src/dice_viewer'
    scientific=('src/dice_viewer/data/','src/dice_viewer/models/',
                'src/dice_viewer/evaluation/','src/dice_viewer/training/')
    for name in manifest['files']:
        if name.startswith(scientific) and name.endswith('.py') and not (root/name).is_file():
            raise ValueError('Approved scientific overlay source is missing')
    for path in package.rglob('*.py'):
        relative=path.relative_to(root).as_posix()
        if 'recognition' in path.relative_to(package).parts:
            continue
        if path == package/'__init__.py':
            body=ast.parse(path.read_text(encoding='utf-8')).body
            if all(isinstance(item,ast.Expr) and isinstance(item.value,ast.Constant)
                   and isinstance(item.value.value,str) for item in body):
                continue  # The current package initializer contains only a docstring.
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'].get(relative):
            raise ValueError('Scientific overlay source differs from approved release')


def compose_command(project, release):
    state=ROOT/'runtime/portable'/project
    return ['docker','compose','--project-name',project,'--project-directory',str(ROOT),
            '--env-file',str(state/'releases'/f'{release.name}.env'),
            '-f',str(release/'deploy/v1/compose.yaml'),
            '-f',str(ROOT/'deploy/observability/compose.yaml')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','status','logs','stop'))
    parser.add_argument('--project',default='dice-observability')
    parser.add_argument('--port',type=int,default=25000)
    args=parser.parse_args()
    if (not args.project or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.project)
            or not 1024<=args.port<=65533):
        parser.error('Use a lowercase Compose project and port 1024..65533')
    try:
        release=model.restore()
        if args.command=='run':
            verify_overlay(ROOT,release)
            subprocess.run([sys.executable,str(ROOT/'tools/model.py'),'run',
                            '--project',args.project,'--port',str(args.port)],check=True)
        state=ROOT/'runtime/portable'/args.project
        logs=state/'observability'
        if args.command=='run':
            logs.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env['DICE_OBSERVABILITY_SOURCE_DIR']=str(ROOT/'src')
        env['DICE_OBSERVABILITY_HOST_DIR']=str(logs)
        command=compose_command(args.project,release)
        operations={'run':['up','-d','--no-build','--wait','--wait-timeout','180','api'],
                    'status':['ps','--all'],'logs':['logs','--tail','100','api'],'stop':['stop']}
        subprocess.run(command+operations[args.command],cwd=ROOT,env=env,check=True)
        if args.command=='run':
            print(f'Instrumented API: http://127.0.0.1:{args.port}; JSONL directory: {logs}',flush=True)
        return 0
    except (OSError,ValueError,subprocess.CalledProcessError):
        print('observability_launch_failed; inspect Docker output and approved-source checks',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
