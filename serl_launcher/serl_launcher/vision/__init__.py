from serl_launcher.vision.resnet_v1 import resnetv1_configs
from serl_launcher.vision.vit import vit_configs

encoders = dict()
encoders.update(resnetv1_configs)
encoders.update(vit_configs)
