import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from modules import Logger
import os
import shutil
import subprocess

def _get_run_name(str):
    config = str.split('/')
    if len(config) == 1: return config[0]
    else: return config[0] + '.' + config[1]

OmegaConf.register_new_resolver('get_run_name', _get_run_name)

def debug():
    if len(sys.argv) < 2:
        raise ValueError('Usage: debug <config_name> [overrides]')

    config_name, overrides = sys.argv[1], sys.argv[2:]
    cmd = [
        sys.executable,
        "-m", "debugpy",
        "--listen", "5678",
        "--wait-for-client",
        "main.py",
        f"--config-name={config_name}",
        *overrides,
    ]
    subprocess.run(cmd)

def run():
    argv = sys.argv
    if len(argv) >= 2:
        config_name, overrides = argv[1], argv[2:]
        sys.argv = [argv[0], f"--config-name={config_name}", *overrides]
    else: 
        raise ValueError('Usage: run <config_name> [overrides]')
    _run()

@hydra.main(version_base=None, config_path='configs', config_name='default')
def _run(config: DictConfig) -> None:
    from procedures import Procedure
    procedure = Procedure.load(config.procedure, config)
    Logger.start(log_dir=config.log_dir)
    procedure()
    Logger.stop()

def clean():
    for name in os.listdir('./logs'):
        path = os.path.join('./logs', name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

if __name__ == '__main__':
    _run()
