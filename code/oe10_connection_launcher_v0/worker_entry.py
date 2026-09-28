"""Owned process-group entry: record actual imports/environment, then exec frozen model CLI."""
import ctypes
import importlib.metadata
import signal
import stat
import json
import os
from pathlib import Path
import platform
import sys


def arm_parent_death(expected_parent):
    # Linux preserves PDEATHSIG across this ordinary, non-setuid Python exec.
    # It is not inherited by arbitrary fork descendants; the fixed model CLI
    # uses a single process with threads. The launcher separately checks PGID.
    if not sys.platform.startswith('linux') or os.getppid() != expected_parent:
        raise RuntimeError('Parent PID changed before death-signal setup')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != expected_parent:
        raise RuntimeError('Parent died during death-signal setup')
    return {'signal':'SIGKILL','expected_parent_pid':expected_parent,
            'scope':'direct fixed Python worker only; fork descendants not covered'}


def main():
    output = Path(sys.argv[1]); expected_parent = int(sys.argv[2]); assert sys.argv[3] == '--'
    parent_guard = arm_parent_death(expected_parent)
    command = sys.argv[4:]; assert command and command[0] == sys.executable
    if os.stat(command[0]).st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise RuntimeError('Credential-changing interpreter forbidden')
    if 'security.capability' in os.listxattr(command[0]):
        raise RuntimeError('File-capability interpreter forbidden')
    receipt = {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
               'pid':os.getpid(),'process_group':os.getpgrp(),'PYTHONPATH':os.environ.get('PYTHONPATH'),
               'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'packages':{},'parent_death_guard':parent_guard}
    try:
        import torch
        import transformers
        receipt.update(torch_import_path=torch.__file__,transformers_import_path=transformers.__file__,
                       torch_cuda_version=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized())
        for name in ('torch','transformers','numpy','safetensors','accelerate'):
            try: receipt['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: receipt['packages'][name] = None
        receipt['status'] = 'imports_passed'
    except BaseException as exc:
        receipt.update(status='imports_failed',error=repr(exc)); raise
    finally:
        with output.open('x') as f:
            json.dump(receipt,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.execve(command[0],command,os.environ.copy())


if __name__ == '__main__':
    main()
