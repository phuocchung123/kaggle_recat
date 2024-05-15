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
    print('multi_head=32,weights for combination R,P and reagents=0.7,epoch=30,dr_out=0.1,non attention, noncross, softmax')

    finetune(args)

    end_point=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('the start point for running is',start_point)

    print('the end point for running is',end_point)
