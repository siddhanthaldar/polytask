import torch.nn as nn
import utils
from .progressive_net import PNN

class Encoderv2(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        frame_size = ((obs_shape[1]-3) // 2 + 1) - 6 # metaworld -84, maniskill2 - 128, kitchen - 256
        self.trunk = nn.Sequential(nn.Linear(32 * frame_size * frame_size, 512),
                						     nn.LayerNorm(512), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        frame_size = ((obs_shape[1]-3) // 2 + 1) - 6 # metaworld -84, maniskill2 - 128, kitchen - 256
        self.trunk = nn.Sequential(nn.Linear(32 * frame_size * frame_size, 512),
                						     nn.LayerNorm(512), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h

class ProgressiveEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        self.conv_layers = [obs_shape[0], 32, 32, 32, 32]
        self.convnet = PNN(len(self.conv_layers)-1, layer_type='conv')

        frame_size = ((obs_shape[1]-3) // 2 + 1) - 6 # metaworld -84, maniskill2 - 128, kitchen - 256

        self.linear_layers = [32 * frame_size * frame_size, 512]
        self.trunk = PNN(len(self.linear_layers)-1, layer_type='linear', no_relu=True)
        self.norm = nn.Sequential(nn.LayerNorm(512), nn.Tanh())
        # self.trunk = nn.Sequential(nn.Linear(32 * frame_size * frame_size, 512),
        #         						     nn.LayerNorm(512), nn.Tanh())

        self.apply(utils.weight_init)

    def new_task(self):
        # freeze existing columns
        self.convnet.freeze_columns()
        self.trunk.freeze_columns()

        # add new columns
        self.convnet.new_task(self.conv_layers)
        self.trunk.new_task(self.linear_layers)
    
    def parameters(self):
        return list(self.convnet.parameters(-1)) + list(self.trunk.parameters(-1))

    def forward(self, obs, task_id=-1):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs, task_id=task_id)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h, task_id=task_id)
        h = self.norm(h)
        return h
