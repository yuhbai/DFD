import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import csv
import script.common as common
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import matplotlib.pyplot as plt
from collections import OrderedDict
import script.metrics as metrics
from torchvision.models import resnet50
from model.vision_transformer import VisionTransformer
import cv2
LOGGER = logging.getLogger(__name__)

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class Discriminator_perlin(nn.Module):
    def __init__(self, embed_dim=1536, depth=2, num_heads=16, patches=1024):
        super().__init__()
        self.ViT = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_patches=patches)
        self.tail = torch.nn.Linear(embed_dim, 1, bias=False)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), 
                                        torch.nn.BatchNorm1d(embed_dim),
                                        torch.nn.LeakyReLU(0.2)
                                        )
        
    def forward(self,x, batchsize):
        x = self.mlp(x)
        _, c = x.shape
        x = x.reshape(batchsize, -1, c)
        x = self.ViT(x)
        x = self.tail(x)
        return x

class Discriminator_gaussian(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator_gaussian, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x
    
class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        self.in_planes = in_planes
       
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):

        return self.layers(x)

class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1


class model(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(model, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,

        lr_perlin=0.0001,
        lr_gaussian=0.0002,
        lr_proj=0.0005,
     
        dsc_hidden=1024,
        epochs = 10,
        noise_std=0.015,
        mix_noise=1,
        noise_type="GAU",
        meta_epochs=15, # 40
        aed_meta_epochs=1,
        gan_epochs=10, 
        n_layers = 2,
        **kwargs,
    ):
        self.device = device
        self.backbone = backbone.to(self.device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
       
        self.epochs = epochs
        self.lr_proj = lr_proj

        self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, 1, 0)
        self.pre_projection.to(self.device)
        self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), self.lr_proj)
       
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std

        self.aed_meta_epochs = aed_meta_epochs
        self.gan_epochs = gan_epochs
        self.meta_epochs = meta_epochs

        self.lr_perlin = lr_perlin
        self.lr_gaussian = lr_gaussian
        self.discriminator_perlin = Discriminator_perlin(embed_dim=self.target_embed_dimension, depth=1, num_heads=16, patches=1024)
        self.discriminator_perlin.to(self.device)
        
        self.discriminator_gaussian = Discriminator_gaussian(self.target_embed_dimension, n_layers=n_layers, hidden=dsc_hidden)
        self.discriminator_gaussian.to(self.device)

        self.opt = torch.optim.Adam(
            [
                {"params":self.discriminator_gaussian.parameters(), "lr": self.lr_gaussian, "weight_decay": 1e-5},
                {"params":self.discriminator_perlin.parameters(), "lr": self.lr_perlin, "weight_decay": 1e-5}
            ]
        )
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, (meta_epochs - aed_meta_epochs) * epochs, self.lr_perlin*.4)
   
    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images):

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features_org = [features[layer] for layer in self.layers_to_extract_from]
        
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features_org
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        return features, patch_shapes


    def train(self, train_data, test_data):
        state_dict = {}

        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
 
        def update_state_dict(d):
            
            state_dict["discriminator_gaussian"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator_gaussian.state_dict().items()})  
            state_dict["discriminator_perlin"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator_perlin.state_dict().items()})
            state_dict["pre_projection"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.pre_projection.state_dict().items()})
        best_record = None
        
        for i_mepoch in range(self.meta_epochs):
            self.train_module(train_data)
            gt_list_sp, pr_list_sp, gt_px, pr_px = self.predict(test_data)
            auroc_sp = metrics.compute_imagewise_retrieval_metrics(pr_list_sp,gt_list_sp,inference=False)["auroc"]
            auroc_px =  metrics.compute_pixelwise_retrieval_metrics(pr_px,gt_px,inference=False)["auroc"]
            if len(gt_px.shape) == 4:
                gt_px = gt_px.squeeze(1)
            if len(pr_px.shape) == 4:
                pr_px = pr_px.squeeze(1)
            aupro = metrics.cal_pro_score(gt_px, pr_px)    
            if best_record is None:
                best_record = [auroc_sp, auroc_px, aupro]
                update_state_dict(state_dict)
            else:
                if auroc_sp + auroc_px + aupro > best_record[0] + best_record[1] + best_record[2]:
                    best_record = [auroc_sp, auroc_px, aupro]
                    update_state_dict(state_dict)
            print(f"----- {i_mepoch} I-AUROC:{round(auroc_sp*100,3)}(MAX:{round(best_record[0]*100,3)})"
                  f"  P-AUROC{round(auroc_px*100,3)}(MAX:{round(best_record[1]*100,3)})"
                  f"  AUPRO{round(aupro*100, 3)}(MAX:{round(best_record[2]*100,3)})-----")

        auroc_sp, auroc_px, aupro = best_record
        
        gt_list_sp, pr_list_sp, gt_px, pr_px = self.predict(test_data)
  
        
        torch.save(state_dict, ckpt_path)
        
        return auroc_px, auroc_sp, aupro
    
    def train_module(self, input_data):
        _ = self.forward_modules.eval()
        self.discriminator_perlin.train()
        self.discriminator_gaussian.train()
        self.pre_projection.train()
        i_iter = 0
        current_time = get_current_time()
        writer = SummaryWriter('./logs/' + current_time + str(self.epochs))
        with tqdm.tqdm(self.epochs) as pbar:
            for i_epochs in range(self.epochs):
                for i_batch, data_item in enumerate(input_data):
                    self.opt.zero_grad()
                    self.proj_opt.zero_grad()
                    mask = data_item["anomaly_mask"].to(self.device)
                    anomaly = data_item["has_anomaly"].to(self.device)
                    lab = data_item["lab"].to(self.device).float()
                    res = data_item["res"].to(self.device).float()
                    lab_aug = data_item["lab_aug"].to(self.device).float()
                    res_aug = data_item["res_aug"].to(self.device).float()

                    
                    features_lab, patch_shape = self._embed(lab)
                    features_res, _ = self._embed(res)
                    features_lab_aug, _ = self._embed(lab_aug)
                    features_res_aug, _ = self._embed(res_aug)


                    b,c,h,w = lab.shape
                    features_res_proj = self.pre_projection(features_res)
                    features_res_aug_proj = self.pre_projection(features_res_aug)
                    features_lab_proj = self.pre_projection(features_lab)
                    features_lab_aug_proj = self.pre_projection(features_lab_aug)

                     
                    scores_lab = self.discriminator_perlin(features_lab_aug_proj, b)
                    scores_res = self.discriminator_perlin(features_res_aug_proj, b)

                    # mask 
                    mask_ = mask.flatten()  
                    down_ratio_y = int(mask.shape[2]/patch_shape[0][0])
                    down_ratio_x = int(mask.shape[3]/patch_shape[0][1])
                    anomaly_mask = torch.nn.functional.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()
                    anomaly_mask = anomaly_mask.reshape(-1, 1)


                    loss_gaussian  = self.cal_loss_gaussian(features_res_proj) + self.cal_loss_gaussian(features_lab_proj)
                    loss_perlin = self.cal_loss_perlin(scores_res, mask_, patch_shape, b, anomaly) + self.cal_loss_perlin(scores_lab, mask_, patch_shape, b, anomaly)
                    loss_sim =  self.sim_loss(features_res_proj, features_res_aug_proj, anomaly_mask) + self.sim_loss(features_lab_proj, features_lab_aug_proj, anomaly_mask)

                    loss = loss_gaussian + loss_perlin + loss_sim 

                    loss.backward()
                    self.proj_opt.step()
                    self.opt.step()

                self.dsc_schl.step()    
                i_iter +=1
                pbar.update(1)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

    def cal_loss_gaussian(self, features_proj):
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([features_proj.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
        noise = torch.stack([
            torch.normal(0, self.noise_std * 1.1**(k), features_proj.shape)
            for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = features_proj + noise
        scores_denoise = self.discriminator_gaussian(torch.cat([features_proj, fake_feats]))
        true_scores = scores_denoise[:len(features_proj)]
        fake_scores = scores_denoise[len(fake_feats):]
        true_loss = torch.clip(-true_scores + 0.5, min=0)
        fake_loss = torch.clip(fake_scores + 0.5, min=0)
        loss_gaussian = true_loss.mean() + fake_loss.mean()
        return loss_gaussian

    def cal_loss_perlin(self, scores_a, mask_, patch_shape, batch_size, anomaly):

        scores_a = scores_a.reshape(batch_size,patch_shape[0][0],patch_shape[0][1]).unsqueeze(1)
        scores_a = F.interpolate(scores_a, size=self.input_shape[1], mode='bilinear', align_corners=False)
        
        scores_a_ = - scores_a.reshape(batch_size, -1)
        scores_a_ = torch.sigmoid(scores_a_)                                    
        scores_cls, _ = torch.max(scores_a_, dim=1)
        loss_l2 = nn.MSELoss()
        loss_cls = loss_l2(scores_cls, anomaly.squeeze(1))

        scores_a = scores_a.flatten() 
        if torch.sum(mask_) == 0:
            normal_loss = torch.clip(0.5 - scores_a, min=0)
            loss_pixel = normal_loss.mean()
        else:
            anomaly_scores = scores_a[mask_ == 1]
            normal_scores = scores_a[mask_ == 0]
            anomaly_loss = torch.clip(0.5 + anomaly_scores, min=0)
            normal_loss = torch.clip(0.5 - normal_scores, min=0)
            loss_pixel = anomaly_loss.mean() + normal_loss.mean() 

        return 0.5*loss_pixel + 0.5*loss_cls 
    
    def sim_loss(self, features_proj, features_aug_proj, mask):
        
        if torch.sum(mask) == 0:
            loss_sim = 0
        else:
            def reshape(features, mask): 
                features = mask*features
                nozero = torch.any(features!=0, dim=1)
                features = features[nozero]
                return features
            features_proj = reshape(features_proj, mask)   
            features_aug_proj = reshape(features_aug_proj, mask)  
            sim = torch.nn.functional.cosine_similarity(features_aug_proj, features_proj)
            loss_sim = sim.mean()

        return loss_sim

    def predict(self, dataloader, path=None):
        if path != None:
            ckpt_path = os.path.join(path, "ckpt.pth")
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.discriminator_gaussian.load_state_dict(state_dict['discriminator_gaussian'])      
                self.discriminator_perlin.load_state_dict(state_dict['discriminator_perlin'])      
                self.pre_projection.load_state_dict(state_dict["pre_projection"])

        _ = self.forward_modules.eval()
        self.pre_projection.eval()
        self.discriminator_perlin.eval()
        self.discriminator_gaussian.eval()
        gt_list_sp = []
        pr_list_sp = []
        pr_noise_list_sp = []
        gt_px= []
        pr_px = []
        pr_noise_px = []
        with torch.no_grad():
            with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
                for i, sample in enumerate(data_iterator):

                    gt = sample["mask"].to(self.device)
                    lab = sample["lab"].to(self.device).float()
                    res = sample["res"].to(self.device).float()
                    features_lab, patch_shape = self._embed(lab)
                    features_res, patch_shape = self._embed(res)
                   
                    features_lab = self.pre_projection(features_lab)
                    features_res = self.pre_projection(features_res)

                    scores_lab = - self.discriminator_perlin(features_lab, 1)
                    scores_res = - self.discriminator_perlin(features_res, 1)
                    scores_lab_noise =  - self.discriminator_gaussian(features_lab)
                    scores_res_noise =  - self.discriminator_gaussian(features_res)
                    anomaly_map =  self.cal_map(scores_res, patch_shape) +self.cal_map(scores_lab, patch_shape) 
                    anomaly_map_noise =  self.cal_map(scores_res_noise, patch_shape) +self.cal_map(scores_lab_noise, patch_shape)   

                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    gt_px.append(gt.squeeze(1).cpu().numpy())
                    pr_px.append(anomaly_map)
                    pr_noise_px.append(anomaly_map_noise)
                    
                    gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                    pr_list_sp.append(np.max(anomaly_map))
                    pr_noise_list_sp.append(np.max(anomaly_map_noise))

            gt_px = np.array(gt_px)
            pr_px = np.array(pr_px)
            pr_noise_px = np.array(pr_noise_px)
 
        pr_list_sp, pr_px = self.normalize(pr_list_sp, pr_px) 
        pr_noise_list_sp, pr_noise_px = self.normalize(pr_noise_list_sp, pr_noise_px) 


        pr_sp = 0.5*pr_list_sp + 0.5*pr_noise_list_sp 
        pr_sp = pr_sp.tolist()
        pr_px = 0.5*pr_px + 0.5*pr_noise_px 

        return gt_list_sp, pr_sp, gt_px, pr_px
    
    def cal_map(self, scores, patch_shape):
        scores = scores.reshape(1,patch_shape[0][0],patch_shape[0][1]).unsqueeze(1)
        scores = F.interpolate(scores, size=self.input_shape[1], mode='bilinear', align_corners=False)
        anomaly_map = scores.reshape(self.input_shape[1],self.input_shape[2]).cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        return anomaly_map
                       
    def normalize(self, pr_list_sp, pr_px):

        pr_list_sp = np.array(pr_list_sp)
        min_sp = pr_list_sp.min(axis=-1).reshape(-1,1)
        max_sp = pr_list_sp.max(axis=-1).reshape(-1,1)
        pr_list_sp = (pr_list_sp - min_sp) / (max_sp - min_sp)
        pr_list_sp = np.mean(pr_list_sp, axis=0)
      
        pr_px = np.array(pr_px)
        min_scores = pr_px.reshape(len(pr_px), -1).min(axis=-1).reshape(-1, 1, 1, 1)
        max_scores = pr_px.reshape(len(pr_px), -1).max(axis=-1).reshape(-1, 1, 1, 1)
        pr_px = (pr_px - min_scores) / (max_scores - min_scores)
        pr_px = np.mean(pr_px, axis=0)

        return pr_list_sp, pr_px
    
    def visual(self, dataloader, path):
        ckpt_path = os.path.join(path, "ckpt.pth")
        if os.path.exists(ckpt_path):
            print('yes')
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.discriminator_gaussian.load_state_dict(state_dict['discriminator_gaussian'])      
            self.discriminator_perlin.load_state_dict(state_dict['discriminator_perlin'])      
            self.pre_projection.load_state_dict(state_dict["pre_projection"])

        _ = self.forward_modules.eval()
        self.pre_projection.eval()
        self.discriminator_perlin.eval()
        self.discriminator_gaussian.eval()
 
        with torch.no_grad():
            with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
                for i, sample in enumerate(data_iterator):
                    image = sample["image"].to(self.device).float()
                    gt = sample["mask"].to(self.device)

                    lab = sample["lab"].to(self.device).float()
                    res = sample["res"].to(self.device).float()
                    features_lab, patch_shape = self._embed(lab)
                    features_res, _ = self._embed(res)
                    
                    features_lab = self.pre_projection(features_lab)
                    features_res = self.pre_projection(features_res)

                    scores_lab = - self.discriminator_perlin(features_lab, 1)
                    scores_res = - self.discriminator_perlin(features_res, 1)
                    scores_lab_noise =  - self.discriminator_gaussian(features_lab)
                    scores_res_noise =  - self.discriminator_gaussian(features_res)
                    anomaly_map = self.cal_map(scores_lab, patch_shape) + self.cal_map(scores_res, patch_shape) 
                    anomaly_map_noise = self.cal_map(scores_lab_noise, patch_shape) + self.cal_map(scores_res_noise, patch_shape)      

                    anomaly_map_final = anomaly_map_noise + anomaly_map
                    a_min, a_max = anomaly_map_final.min(), anomaly_map_final.max()
                    gray = (anomaly_map_final-a_min)/(a_max - a_min) 
                    gray = gray.reshape((256,256))
                    plt.subplot(121)
                    plt.imshow(gray)
                    plt.show()
                    # gray = cv2.cvtColor(np.transpose(pr_px,(1,2,0)), cv2.COLOR_GRAY2BGR)
                    heatmap = cv2.applyColorMap(np.uint8(gray*255), cv2.COLORMAP_JET)
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    mask = gt.cpu().numpy()
                    mask = mask.reshape((256,256))
                    cv2.imwrite(os.path.join('visual', 'heatmap.jpg'), heatmap)
                    cv2.imwrite(os.path.join('visual', 'mask.jpg'), mask * 255)
                    print('over')

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )

        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


def create_storage_folder(
    main_folder_path, project_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path

def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


def set_torch_device(gpu_ids):

    return torch.device("cuda:{}".format(gpu_ids))

from datetime import datetime

def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time

