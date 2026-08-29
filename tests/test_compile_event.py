from pathlib import Path
import sys
from fable.language import load_event, compile_event
DEFAULT_CE=Path('ce_definitions/brians_ce.yaml')
def main()->int:
    path=Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT_CE
    print(f'CE: {path}')
    print('\n[1/2] PARSE')
    try:event=load_event(path)
    except (ValueError,FileNotFoundError) as exc:
        print('PARSE: FAILED');print(exc);print('\n[2/2] COMPILE\nCOMPILE: SKIPPED');return 1
    print('PARSE: SUCCESS');print(f'  event={event.name}');print(f'  root={event.pattern.op}')
    print('\n[2/2] COMPILE')
    try:compile_event(event)
    except ValueError as exc:
        print('COMPILE: FAILED');print(exc);return 1
    print('COMPILE: SUCCESS');return 0
if __name__=='__main__':raise SystemExit(main())
