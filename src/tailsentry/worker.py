"""Fixed-operation subprocess for one bounded local estimate."""
import json
import sys
from .service import investigate


def main():
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('Request too large')
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError('Configuration must be an object')
        print(json.dumps({'record': investigate(config)}, allow_nan=False))
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        print(json.dumps({'error': str(error)}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
