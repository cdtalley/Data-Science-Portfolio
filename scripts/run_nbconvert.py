"""
Run nbconvert --execute on a notebook with Windows-friendly settings.

On Windows, Python 3.8+ uses ProactorEventLoop by default. Jupyter/ZMQ need
add_reader support, which causes RuntimeWarning and can hang kernel replies.
This script sets WindowsSelectorEventLoopPolicy *before* any ZMQ import so
nbconvert runs without the warning and without timeouts from a stuck kernel.

Usage (from repo root):
  python scripts/run_nbconvert.py path/to/notebook.ipynb
  python scripts/run_nbconvert.py path/to/notebook.ipynb --timeout 1800
"""
import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Non-interactive backend so plotting never blocks or opens windows
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

# Now safe to run nbconvert (pass through all args except our script name)
from nbconvert.nbconvertapp import main

if __name__ == "__main__":
    # nbconvert expects argv[0] to be 'nbconvert'
    sys.argv = ["nbconvert"] + sys.argv[1:]
    main()
