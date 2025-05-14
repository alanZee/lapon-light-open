import os, sys
from pathlib import Path
ROOT = os.path.abspath("../lapon") # LaPON root directory
if str(ROOT) not in sys.path:
    sys.path.append(ROOT) # add ROOT to PATH for import
from engine.base_trainer import BaseTrainer


if __name__ == '__main__':
    cfg = Path(ROOT) / 'cfg/default.yaml'
    assert Path(cfg).exists(), f'ERROR ❌ the cfg path ({cfg}) is wrong'
    
    trainer = BaseTrainer(
        cfg=cfg
        )
    
    # trainer.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    # trainer.model.no_ni = False # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
    # trainer.model.eval()
    # torch.autograd.set_detect_anomaly(True) # for debug

    # trainer.cfg['epochs'] = 10
    # trainer.cfg['optimizer'] = 'Adam' 
    # trainer.cfg['lr0'] = 1e-5
    
    trainer.fit()
    # print(trainer.model.training)
    print("✅ finish!")

