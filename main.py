import os
import torch
import numpy as np 
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import logging
import model.utils as utils
from data.data_loader import MVTecTrainDataset,MVTecTestDataset, VisATrainDataset, VisATestDataset
import json
import random
import script.backbones as backbones

LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--backbone_name", type=str, default="wideresnet50")
    parser.add_argument("--layers_to_extract_from", "-le", type=str, default=["layer2","layer3"])
  

    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--patchoverlap", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=4, help="train epochs")
    parser.add_argument("--meta_epochs", type=int, default=20, help="train")
    parser.add_argument("--n_layers", type=int, default=2, help="layers of discriminator")
    parser.add_argument("--lr_perlin", type=float, default=0.0001, help="dis_perlin lr")
    parser.add_argument("--lr_gaussian", type=float, default=0.0002, help="dis_gaussian lr")
    parser.add_argument("--lr_proj", type=float, default=0.0005, help="proj lr")
    
    #save_path
    parser.add_argument("--save_path", type=str, default="2_shot")
    #dataset
    parser.add_argument('--dataset_name', action='store', type=str, default='visa')
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='../datasets/dtd/images/')
    parser.add_argument('--data_path',type=str, default="/opt/data/private/datasets/VisA/")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--imagesize", default=256, type=int)
    parser.add_argument('--k_shot',default=1, type=int)
    parser.add_argument('--num',default=80, type=int, help='number of augmented images')
    return parser.parse_args() 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_infer(args):
    if args.dataset_name == 'visa':
        classes = ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'candle', 'pipe_fryum', 'capsules', 
        'cashew', 'chewinggum', 'fryum','macaroni1', 'macaroni2',] 
    if args.dataset_name == 'mvtec':
        classes = ["bottle", "cable", "capsule", "carpet", "grid","hazelnut", "leather", "metal_nut", "pill", 
                "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

    with open("./data/anomaly_mask.json") as f:
        anomaly_mask = json.load(f)

    result_collect = []
    run_save_path = utils.create_storage_folder('results',args.save_path, mode="iterate")
    models_dir = os.path.join(run_save_path, "models")
    i = 0
    for _class_ in  classes:
        print("processing:{}/{}".format(i+1,len(classes)))
        print("current class:",_class_)
        i += 1
        if args.dataset_name == 'mvtec':
            bg_re = anomaly_mask[_class_]["bg_reverse"]
            use_mask = anomaly_mask[_class_]["use_mask"]
            train_path = args.data_path + _class_ +'/train/good/'
            test_path = args.data_path + _class_ + '/test/'
            train_data = MVTecTrainDataset(train_path, args.anomaly_source_path, resize_shape=args.imagesize, k_shot=args.k_shot, num=args.num, use_mask=use_mask, bg_reverse=bg_re)
            test_data = MVTecTestDataset(test_path,resize_shape=args.imagesize)
        if args.dataset_name == 'visa':
            train_path = args.data_path + _class_ +'/train/good/'
            test_path = args.data_path + _class_ + '/test/'
            train_data = VisATrainDataset(train_path, args.anomaly_source_path, resize_shape=args.imagesize, k_shot=args.k_shot, num=args.num)
            test_data = VisATestDataset(test_path,resize_shape=args.imagesize)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)
       
        device = utils.set_torch_device(args.gpu)
        net = backbones.load(args.backbone_name)
        DFD = utils.model(device)
        DFD.load(
            device=device,
            backbone=net,
            layers_to_extract_from=args.layers_to_extract_from, 
            input_shape=(3, args.imagesize, args.imagesize),
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            patchstride=args.patchstride,
            lr_perlin=args.lr_perlin,
            lr_gaussian=args.lr_gaussian,
            lr_proj=args.lr_proj,
            dsc_hidden=args.dsc_hidden,
            epochs=args.epochs,
            meta_epochs=args.meta_epochs,      
            )
        DFD.set_model_dir(models_dir, _class_)
        auroc_px, auroc_sp, pro_auc = DFD.train(train_dataloader, test_dataloader)
        result_collect.append(
                {
                    "dataset_name": _class_,
                    "sample_auroc": round(auroc_sp*100,3),
                    "pixel_auroc" :round(auroc_px*100,3),
                    "aupro" :round(pro_auc*100,3),
                }
        )

        for key, item in result_collect[-1].items():
            if key != "dataset_name":
                LOGGER.info("{0}: {1:3.3f}".format(key, item))
                print("{0}: {1:3.3f}".format(key, item))
    result_scores = [list(results.values())[1:] for results in result_collect]
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


if __name__ == '__main__':

    
    args = parse_args()
    setup_seed(args.seed)
    train_and_infer(args)
