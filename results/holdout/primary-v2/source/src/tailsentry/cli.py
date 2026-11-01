"""Programmatic engine and loopback viewer CLI."""
import argparse,json
from pathlib import Path
from .service import investigate,serve


def main():
    parser=argparse.ArgumentParser(description='TAILSENTRY rare-event research engine')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('estimate');run.add_argument('--case',default='delta_gamma_1e4');run.add_argument('--threshold',type=float)
    run.add_argument('--budget',type=int,default=16384);run.add_argument('--seed',type=int,default=42)
    run.add_argument('--confidence',type=float,default=.95);run.add_argument('--precision',type=float,default=.1)
    run.add_argument('--method',default='envelope_projected_fixed');run.add_argument('--output',type=Path)
    view=commands.add_parser('serve');view.add_argument('--port',type=int,default=8768)
    args=parser.parse_args()
    if args.command=='serve':return serve(args.port)
    config={k:v for k,v in vars(args).items() if k not in ('command','output') and v is not None}
    result=investigate(config);text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as file:file.write(text+'\n')
        print(args.output.resolve())
    else:print(text)

if __name__=='__main__':main()
