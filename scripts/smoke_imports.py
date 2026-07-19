import importlib
import sys
import os

# Ensure project root is on sys.path so packages are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print('cwd=', os.getcwd())
print('sys.path[0]=', sys.path[0])

modules=['data','models','evaluation','utils']
for m in modules:
    try:
        importlib.import_module(m)
        print(f'Imported {m}')
    except Exception as e:
        print(f'Failed importing {m}: {e}')
        sys.exit(1)
print('All imports OK')
