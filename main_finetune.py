import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from rdkit import rdBase

from src_chung.get_reaction_data import get_graph_data
from src_chung.finetune import finetune
import warnings
import datetime

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

warnings.filterwarnings('ignore') 

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--graph_save_path", type=str, default="/kaggle/working/sample/data_chung/"
    )

    arg_parser.add_argument("--seed", type=int, default=27407)

    args = arg_parser.parse_args()


    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    if not os.path.exists("/kaggle/working/sample/data_chung/model/finetuned/"):
        os.makedirs("/kaggle/working/sample/data_chung/model/finetuned/")

    start_point=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('crossattention, numlayer=1,multi_head=8, epoch=50, dim_inner=2048,d_k=64,d_v=64')
    print('commit: modify SubLayer.py because self attention for each reaction not for batch')

    finetune(args)

    end_point=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('the start point for running is',start_point)

    print('the end point for running is',end_point)
