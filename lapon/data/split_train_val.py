# LaPON, GPL-3.0 license
# Dataloaders and dataset utils
# coding:utf-8

import os, glob
from pathlib import Path
import random
import argparse
import numpy as np

random.seed(0)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def main(opt):
    shuffle = opt.shuffle
    datadir = opt.datadir
    train, val = opt.train, opt.val
    assert train + val <= 1.0, f"train({train}) + val({val}) must <= 1.0"
    
    if os.sep in datadir:
        ls = datadir.split(os.sep)
    else:
        ls =  datadir.split("/")
    last_dir = ls[-1] if ls[-1] != '' else ls[-2]
    
    p = Path(datadir)  # os-agnostic
    save_dir = str(p.parent)
    save_train = os.path.join(save_dir, f'train_{last_dir}.txt')
    save_val = os.path.join(save_dir, f'val_{last_dir}.txt')
    save_test = os.path.join(save_dir, f'test_{last_dir}.txt')
    
    
    frame_list = glob.glob(os.path.join(datadir, '*'))
    frame_list.sort()
    frame_list = [os.path.abspath(fi) for fi in frame_list]
    # frame_list.sort(key=lambda x: os.path.getmtime(x)) # 使用 os.path.getmtime 获取最后修改时间，并将其转换为元组 (修改时间, 文件名)
    if '\\' in frame_list[0]: # WindowsPath
        frame_list = [frame.replace('\\', '/') for frame in frame_list]
        # frame_list = list(map(lambda frame: frame.replace('\\', '/'), frame_list))
    
    if shuffle:
        random.shuffle(frame_list)
    train_list = frame_list[:int(len(frame_list) * train)]
    val_list   = frame_list[int(len(frame_list) * train):
                            int(np.ceil(len(frame_list) * (train + val)))]
    test_list  = frame_list[int(np.ceil(len(frame_list) * (train + val))):]
    print("train, val, test(number of samples): " + colorstr('green', 'bold', f'{len(train_list), len(val_list), len(test_list)}'))
    
    # save
    train_list = [frame+'\n' for frame in train_list]
    with open(save_train, 'w') as fi:
        fi.writelines(train_list)
    
    val_list = [frame+'\n' for frame in val_list]
    with open(save_val, 'w') as fi:
        fi.writelines(val_list)
    
    test_list = [frame+'\n' for frame in test_list]
    with open(save_test, 'w') as fi:
        fi.writelines(test_list)
    
    print("✅ " + colorstr('green', 'bold', 'underline', 'finish!'))
    # WARNING ⚠️ 
    # ERROR ❌

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/demo/isotropic1024coarse/datadir_u_z1/')
    parser.add_argument('--train', type=float, default=0.8, help="tain val test split ratio")
    parser.add_argument('--val', type=float, default=0.1, help="tain val test split ratio")
    parser.add_argument('--shuffle', type=bool, default=False, help="shuffle file list")
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run(**kwargs):
    """ Usage: import split_train_val; split_train_val.run(datadir='datadir_u_z1', train=0.8, val=0.2) """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

# test: run(datadir="../data/demo/isotropic1024coarse/datadir_u_z1/")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
