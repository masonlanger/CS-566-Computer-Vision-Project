import hydra
from omegaconf import DictConfig, OmegaConf
from modules import Logger

def _get_run_name(str):
    config = str.split('/')
    if len(config) == 1:
        return config[0]
    else:
        return config[0] + '.' + config[1]

OmegaConf.register_new_resolver('get_run_name', _get_run_name)

@hydra.main(version_base=None, config_path='configs', config_name='default')
def run(config: DictConfig) -> None:
    
    match config.procedure:
        case 'generate_observations':
            from procedures.generate_observations import GenerateObservations
            procedure = GenerateObservations(config)
        case 'em':
            from procedures.em import EM
            procedure = EM(config)
        case _: 
            raise ValueError()

    Logger.start(log_dir=config.log_dir)
    procedure()
    Logger.stop()

if __name__ == "__main__": 
    run()