"""Programmatic engine and loopback viewer CLI."""
import argparse,json,os,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from .service import investigate,serve


def _save_estimate(path, text):
    published_directory = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.tailsentry-', dir=path.parent) as directory:
            staged = Path(directory) / 'estimate.json'
            staged.write_text(text+'\n', encoding='utf-8')
            try:
                os.link(staged, path)
            except FileExistsError:
                raise SystemExit(f'Output already exists: {path}. Choose a new path.') from None
            published_directory = directory
    except OSError as error:
        if published_directory is None:
            raise SystemExit(f'Could not save estimate to {path}: {error}') from None
        print(f'Estimate saved to {path}; could not clean staging directory {published_directory}: {error}',
              file=sys.stderr)


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
        _save_estimate(args.output, text)
        print(args.output.resolve())
    else:print(text)

if __name__=='__main__':main()
