"""Programmatic engine and loopback viewer CLI."""
import argparse,json,os,sys
from pathlib import Path
from tempfile import TemporaryDirectory
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
        try:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with TemporaryDirectory(prefix='.tailsentry-', dir=args.output.parent) as directory:
                staged = Path(directory) / 'estimate.json'
                staged.write_text(text+'\n', encoding='utf-8')
                try:
                    os.link(staged, args.output)
                except FileExistsError:
                    raise SystemExit(f'Output already exists: {args.output}. Choose a new path.') from None
        except OSError as error:
            print(f'Could not save estimate to {args.output}: {error}', file=sys.stderr)
            raise SystemExit(1) from None
        print(args.output.resolve())
    else:print(text)

if __name__=='__main__':main()
