from .layer import Layer
from .convolution import *
from .connected import *

class avgpool_layer(Layer):
    pass

class crop_layer(Layer):#只有初始化Layer操作
    pass

class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class softmax_layer(Layer):
    def setup(self, groups):
        self.groups = groups

class dropout_layer(Layer):
    def setup(self, p):
        self.h['pdrop'] = dict({
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': ()
        })

class route_layer(Layer):
    def setup(self, routes):
        self.routes = routes

class reorg_layer(Layer):
    def setup(self, stride):
        self.stride = stride

"""
Darkop Factory
工厂模式，映射到对应类
"""

darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'select': select_layer,
    'route': route_layer,
    'reorg': reorg_layer,
    'conv-select': conv_select_layer,
    'conv-extract': conv_extract_layer,
    'extract': extract_layer
}

def create_darkop(ltype, num, *args):
    op_class = darkops.get(ltype, Layer)#获取对应类，默认为Layer类
    return op_class(ltype, num, *args)#新建一个对应对象，参数：type、num、传入到Layer.setup函数的参数