
from rich.pretty import pretty_repr
import os
import matplotlib.pyplot as plt
import csv
from typing import Dict, Any
import threading
import time
# import wandb as WB
import logging
from rich.logging import RichHandler
from omegaconf import OmegaConf, DictConfig

class Logger:
    '''
    Handles storage of data and metrics.
    1. Logs metrics locally.
    2. Logs metrics remotely using W&B (optional).
    3. Logs to console.
    4. Saves console logs to file.
    '''
    # Class variables
    logger = None
    log_dir = './'
    curr_dir = './'
    write_every = 5
    metric_buffer = {}
    _running = False
    _log_thread = None
    _metrics_lock = threading.Lock()
    _files_lock = threading.Lock()
    wb = None
    
    @classmethod
    def __init__(cls):
        console_handler = RichHandler(
            level=logging.DEBUG,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            log_time_format="%H:%M:%S"
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.addHandler(console_handler)
        cls.logger.propagate = False 

    @classmethod
    def start(cls, log_dir='./', write_every=5):
        '''
        Creates run dir., sets up local metric and console logging.
        '''
        # initialize log dir
        cls.log_dir = log_dir
        os.makedirs(cls.log_dir, exist_ok=True)
        cls.curr_dir = cls.log_dir

        # local metric logging
        cls.write_every = write_every
        cls.metric_buffer: Dict[str, Any] = {}
        cls._log_thread = None
        cls._metrics_lock = threading.Lock()
        cls._files_lock = threading.Lock()

        # start metric thread
        cls._running = True
        cls._log_thread = threading.Thread(target=cls._writer_thread, daemon=True)
        cls._log_thread.start()

        # remote metric logging (wandb)
        cls.wb = None

        # save console logs to file
        file_handler = logging.FileHandler(
            filename=os.path.join(log_dir, "console.log"),
            mode="a"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%H:%M:%S"    # <-- this forces time-only
        ))
        cls.logger.addHandler(file_handler)
        cls.info(f'Logging to {log_dir}.')

    @classmethod
    def use_wb(cls, run):
        cls.wb = run

    @classmethod
    def format_msg(cls, args):
        formatted_args = []
        for arg in args:
            match arg:
                case DictConfig():
                    # convert to dict 
                    dict_arg = OmegaConf.to_container(arg, resolve=True)
                    formatted_args.append(pretty_repr(dict_arg))
                case dict():
                    formatted_args.append(pretty_repr(arg))
                case _:
                    formatted_args.append(str(arg))
        return "\n".join(formatted_args)
    
    @classmethod
    def debug(cls, *args, stacklevel=2, **kwargs): 
        msg = cls.format_msg(args)
        # msg = f"[italic]{msg}[/italic]"
        cls.logger.debug(msg, stacklevel=stacklevel, **kwargs)
        return cls
    
    @classmethod
    def info(cls, *args, stacklevel=2, **kwargs): 
        cls.logger.info(cls.format_msg(args), stacklevel=stacklevel, **kwargs)
        return cls

    @classmethod
    def warning(cls, *args, stacklevel=2, **kwargs): 
        cls.logger.warning(cls.format_msg(args), stacklevel=stacklevel, **kwargs)
        return cls
    
    @classmethod
    def error(cls, *args, stacklevel=2, **kwargs): 
        cls.logger.error(cls.format_msg(args), stacklevel=stacklevel, **kwargs)
        return cls
    
    @classmethod
    def critical(cls, *args, stacklevel=2, **kwargs): 
        cls.logger.critical(cls.format_msg(args), stacklevel=stacklevel, **kwargs)
        return cls
    
    # metric logging
    @classmethod
    def log_metrics(cls, metrics: Dict[str, Any]):
        # local
        for name, value in metrics.items():
            if value is None: continue
            if name not in cls.metric_buffer:
                cls.metric_buffer[name] = []
            
            cls.metric_buffer[name].append(value)

        # remote
        if cls.wb is not None:
            for name, value in metrics.items():
                if value is None: continue
                if name not in cls.metric_steps:
                    cls.metric_steps[name] = 0
                
                cls.run.log({ name: value }, step=cls.metric_steps[name])
                cls.metric_steps[name] += 1

        return cls
    
    @classmethod
    def _writer_thread(cls):
        while cls._running:
            time.sleep(cls.write_every)
            items_to_write = list(cls.metric_buffer.items())
            cls.metric_buffer.clear()
            for name, values in items_to_write:
                cls._write_metric(name, values)

    @classmethod
    def _write_metric(cls, name: str, values: list):
        with cls._files_lock:
            path = os.path.join(cls.log_dir, f'metrics/{name}.csv')
            dir = os.path.dirname(path)
            if not os.path.exists(dir): os.makedirs(dir)
            with open(path, 'a', newline='') as file:
                writer = csv.writer(file)
                for value in values: 
                    if isinstance(value, (list, tuple)):
                        writer.writerow(value) 
                    else:
                        writer.writerow([value])

    
    @classmethod
    def stop(cls) -> None:
        '''Stops the thread that writes metrics to file and W&B if it is being used.'''
        if cls.wb is not None: cls.wb.finish()
        if cls._running:
            cls._running = False
            if cls._log_thread:
                cls._log_thread.join()
    
    @classmethod
    def __enter__(cls):
        cls.start()
        return cls
    
    @classmethod
    def __exit__(cls, exc_type, exc_val, exc_tb):
        cls.stop()
    
    # local data logging
    @classmethod
    def cd(cls, path):
        if path.startswith("~/"):
            cls.curr_dir = os.path.abspath(os.path.join(cls.log_dir, path[2:]))
        elif path == "~":
            cls.curr_dir = cls.log_dir
        else:
            cls.curr_dir = os.path.abspath(os.path.join(cls.curr_dir, path))
        return cls


    @classmethod 
    def save_fig(cls, fig, path: str):
        '''
        Saves matplotlib figure as .png.
        '''
        import matplotlib
        assert isinstance(fig, matplotlib.figure.Figure)

        full_path = os.path.join(cls.log_dir, path)
        dir_path = os.path.dirname(full_path)
        if dir_path: 
            os.makedirs(dir_path, exist_ok=True)

        fig.savefig(full_path, bbox_inches='tight')
        plt.close(fig)
        cls.debug(f'Saved fig. to {full_path}.png.')
        return cls
    
    @classmethod 
    def save_anim(cls, anim, path: str):
        import matplotlib
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

        full_path = os.path.join(cls.log_dir, path)
        dir_path = os.path.dirname(full_path)
        if dir_path: 
            os.makedirs(dir_path, exist_ok=True)

        anim.save(full_path, writer='ffmpeg', fps=10)
        cls.debug(f'Saved animation to {full_path}.')
        return cls

Logger.__init__()