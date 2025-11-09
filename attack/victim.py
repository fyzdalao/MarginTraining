import numpy as np
import torch
from torch import nn
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networks
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        self.arch = args.arch
        self.device = torch.device(args.device)

        self.get_cnn()

        self.batch_size = args.batch_size

        if args.dataset == 'cifar10':
            self.mean = np.reshape([0.49139968, 0.48215827, 0.44653124], [1, 3, 1, 1])
            self.std = np.reshape([0.24703233, 0.24348505, 0.26158768], [1, 3, 1, 1])
        else:
            warnings.warn('Unknown dataset, mean and std are not computed')


    def get_cnn(self):
        num_classes = 10  # 根据数据集设置类别数
        mode = ''  # 根据需要设置mode
        weight = None  # 根据需要设置weight

        # 创建模型
        model = networks.__dict__[self.arch](num_classes=num_classes, mode=mode, weight=weight)
        #checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint_server', self.args.checkpoint,'ckpt.best.pth.tar')  # 参数文件路径
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint_server', self.args.checkpoint,
                                       'ckpt_ep250.best.pth.tar')  # 参数文件路径
        #checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint_server', self.args.checkpoint)

        # 加载参数
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as err:
            err_msg = str(err)
            if 'Missing key(s) in state_dict: "linear.bias"' in err_msg:
                # 重建使用 norm 模式的最后一层（无 bias）
                model = networks.__dict__[self.arch](num_classes=num_classes, mode='norm', weight=weight)
                model.load_state_dict(state_dict)
            else:
                raise
        self.cnn = model.to(self.device).eval()


    def predict(self, x, model, batch_size, device):
        if isinstance(x, np.ndarray):
            x = (x - self.mean) / self.std
            x = x.astype(np.float32)

            batch_amount = math.ceil(x.shape[0] / batch_size)
            batch_logits = []
            with torch.no_grad():
                for i in range(batch_amount):
                    x_now = torch.as_tensor(x[i * batch_size: (i + 1) * batch_size], device=device, dtype=torch.float32)
                    batch_logits.append(model(x_now).detach().cpu().numpy())
            logits = np.vstack(batch_logits)
            return logits
        else:
            x = (x - torch.as_tensor(self.mean, device=device)) / torch.as_tensor(self.std, device=device)
            return model(x)


    def forward(self, x):
        return self.predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)


