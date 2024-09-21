from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from reconstruction_dataset import ReconstructionDataset
import argparse
from omegaconf import OmegaConf

def arg_parser():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        'config_path',
        type=str,
        help='config path')
    parser.add_argument(
        '--resume-path',
        type=str,
        default='models/control_sd21_ini_c64.ckpt',
        help='resume path')
    parser.add_argument(
        '--batch-size', # HACK: batch_size=4 will cause OOM.
        type=int,
        default=2, 
        help='batch size')
    parser.add_argument(
        '--logger_freq',
        type=int,
        default=1000, 
        help='logger logging frequency')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5, 
        help='learning rate')
    parser.add_argument(
        '--gpus',
        type=int,
        default=8, 
        help='num of gpu')
    parser.add_argument(
        '--workers',
        type=int,
        default=4, 
        help='num of workers')
    parser.add_argument(
        '--debug', 
        action='store_true', 
        default=False,
        help='do not log')
    args = parser.parse_args()
    return args
    
def main(args):
    # Configs
    resume_path = args.resume_path
    config_path = args.config_path
    batch_size = args.batch_size  
    logger_freq = args.logger_freq
    learning_rate = args.learning_rate

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    # model = create_model('./models/decoder_ldm.yaml').cpu()
    model = create_model(config_path=config_path)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    # model.sd_locked = sd_locked
    # model.only_mid_control = only_mid_control


    # Misc
    # dataset = MyDataset('/HDD_DISK/users/huangze/All_in_occ/prompt.json')
    main_config = OmegaConf.load(config_path)
    dataset = instantiate_from_config(main_config.data)
    dataloader = DataLoader(dataset, num_workers=args.workers, batch_size=batch_size, shuffle=True, pin_memory=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    # trainer = pl.Trainer(gpus=2, precision=16, callbacks=[logger])
    # trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=args.gpus, 
    #                      precision=16, callbacks=[logger]) 
    if args.debug:
        trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=args.gpus, 
                            precision=16) 
    else:
        trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=args.gpus, 
                         precision=16, callbacks=[logger]) 


    # Train!

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    args = arg_parser()
    main(args)