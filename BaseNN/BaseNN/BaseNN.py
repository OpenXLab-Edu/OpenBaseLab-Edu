import os
import torch
import torch.nn 
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from torch.utils.data import DataLoader
from PIL import Image
import cv2



warnings.filterwarnings('ignore')

def pth_info(checkpoint):
    ckpt = torch.load(checkpoint,map_location='cpu')
    meta = ckpt['meta']
    m = list(meta.keys())
    if 'device' in m:
        m.insert(0,m.pop(m.index('device')))
    if 'backbone' in m:
        m.insert(0,m.pop(m.index('backbone')))
    if 'tool' in m:
        m.insert(0,m.pop(m.index('tool')))

    print(checkpoint,"相关信息如下:")
    print("="*(len(checkpoint)+9))
    for i in m:
        if i == 'word2idx': # 省略完整词表，只展示前10个词
            w2i = '{'
            for j in list(meta[i].keys())[:10]:
                w2i += str(j)+":"+str(meta[i][j])+","
            w2i += '...}'
            print(i,":",w2i)
        else:          
            print(i,":",meta[i])
    print("="*(len(checkpoint)+9))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(torch.nn.Module):
    expansion = 1 #expansion是BasicBlock和Bottleneck的核心区别之一
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if downsample is not None:
            self.downsample = torch.nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else: 
            self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out

class Bottleneck(torch.nn.Module):
    expansion = 4 #expansion是BasicBlock和Bottleneck的核心区别之一
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
        return out

class lstm(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, num_layers=num_layers)
        self.linear = torch.nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden = None):
        seq_len, batch_size = input.size()
        
        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embedding(input)
#         print("embeds",embeds.shape)
        output, hidden = self.lstm(embeds, (h_0, c_0))
#         print("lstm",output.shape)
        output = self.linear(output.view(seq_len * batch_size, -1))
#         print("linear",output.shape,seq_len)

        return output, hidden

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.contiguous().view(x.shape[0], -1)

class Part0(torch.nn.Module):
    def __init__(self, *args):
        super(Part0, self).__init__()
        self.shape = args

    def forward(self, x):
        # print("Part0", type(x), type(x[0]))
        return x[0]
    
class Unsqueeze(torch.nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        # print("unsqueeze",type(x), type(x[:, -1, :].unsqueeze(1)))
        return x[:, -1, :].unsqueeze(1)
    
class Squeeze(torch.nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(1)

class dim2(torch.nn.Module):
    def __init__(self, *args):
        super(dim2, self).__init__()

    def forward(self, x):
        return  x[:, -1, :]

def cal_accuracy(y, pred_y):
    res = pred_y.argmax(axis=1)
    tp = np.array(y.cpu())==np.array(res.cpu())
    acc = np.sum(tp)/ y.shape[0]
    return acc

class ResNet(torch.nn.Module):
 
    def __init__(self, block, layers, size,stride=1, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.stride = stride
        self.inplanes = size[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, size[1], layers,stride=self.stride)
 
        # 初始化权重
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        #         torch.nn.init.constant_(m.weight, 1)
        #         torch.nn.init.constant_(m.bias, 0)
 
 
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # self.inplanes = self.inplanes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return torch.nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.layer1(x) 
        return x
    
    
class nn:
    def __init__(self, task="cls",save_fold=None):
        self.task_type = task.lower()
        self.model = torch.nn.Sequential()
        self.batchsize = None
        self.layers = []
        self.layers_num = 0
        self.optimizer = 'Adam'
        self.x = None
        self.y = None
        self.res = None
        self.save_fold = "checkpoints"
        self.rnn = False
        self.input_size = None
        self.color = None # 若载入图片数据集，该变量表示灰度/RGB等颜色通道信息
        self.transform = None # 若载入图片数据集，该变量记录数据预处理操作
        self.diffusion_model = False
        self.scale = 255.0
        self.last_channel = None
        if save_fold != None:
            self.save_fold = save_fold
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add(self, layer=None, activation=None, optimizer=None, **kw):
        self.layers_num += 1
        self.layers.append(layer)
        
        stride = 1 if 'stride' not in kw.keys() else kw['stride']
        if 'padding' not in kw.keys():
            kw['padding'] = 0

        if layer is not None and isinstance(layer,str):
            layer = layer.lower()
        islayer = [True,True,True]
        if layer == 'linear':
            self.model.add_module('reshape', Reshape(self.batchsize))
            self.model.add_module('linear' + str(self.layers_num), torch.nn.Linear(kw['size'][0], kw['size'][1]))
            self.last_channel = kw['size'][1]
            print("增加全连接层，输入维度:{},输出维度：{}。".format(kw['size'][0], kw['size'][1]))
        elif layer == 'reshape':
            self.model.add_module('reshape', Reshape(self.batchsize))
        elif layer == 'conv2d':
            self.model.add_module('conv2d' + str(self.layers_num), torch.nn.Conv2d(kw['size'][0], kw['size'][1], kw['kernel_size'],stride=stride,padding=kw['padding']))
            print("增加二维卷积层，输入维度:{},输出维度：{},kernel_size: {} ".format(kw['size'][0], kw['size'][1], kw['kernel_size']))
        elif layer == 'conv1d':
            self.model.add_module('conv1d' + str(self.layers_num), torch.nn.Conv1d(kw['size'][0], kw['size'][1], kw['kernel_size'],stride=stride,padding=kw['padding']))
            print("增加一维卷积层，输入维度:{},输出维度：{},kernel_size: {} ".format(kw['size'][0], kw['size'][1], kw['kernel_size']))
        elif layer == 'maxpool':
            stride = None if 'stride' not in kw.keys() else kw['stride']
            self.model.add_module('maxpooling' + str(self.layers_num), torch.nn.MaxPool2d(kw['kernel_size'],stride=stride,padding=kw['padding']))
            print("增加最大池化层,kernel_size: {} ".format(kw['kernel_size']))
        elif layer == 'avgpool':
            stride = None if 'stride' not in kw.keys() else kw['stride']
            self.model.add_module('avgpooling' + str(self.layers_num), torch.nn.AvgPool2d(kw['kernel_size'],stride=stride,padding=kw['padding']))
            print("增加平均池化层,kernel_size: {} ".format(kw['kernel_size']))
        elif layer == 'dropout':
            p = 0.5 if 'p' not in kw.keys() else kw['p']
            self.model.add_module('dropout' + str(self.layers_num), torch.nn.Dropout(p=p) )
            print("增加dropout层,参数置零的概率为: {} ".format(p))
        elif layer =='em_lstm':
            self.rnn = True
            self.vs = len(self.word2idx)
            self.ed = kw['size'][0]
            self.hd = kw['size'][1]
            self.nl = kw['num_layers']
            # self.model.add_module('lstm', lstm(vs,ed,hd, nl))
            # self.model.add_module('LSTM' + str(self.layers_num), torch.nn.LSTM(kw['size'][0], kw['size'][1], kw['num_layers']))
            print("增加em_lstm层, 输入维度：{}, 输出维度：{}, 层数： {} ".format(kw['size'][0], kw['size'][1], kw['num_layers']))
            self.model =  lstm(self.vs,self.ed,self.hd, self.nl)
            self.model.to(self.device)
            self.config = {'model':'lstm','para':{'vs':self.vs,'ed':self.ed,'hd':self.hd,'nl':self.nl}}
        elif layer == 'lstm':
            l = torch.nn.LSTM(kw['size'][0], kw['size'][1], batch_first=True, bidirectional=False)
            l = l.to(self.device)
            self.model.add_module('lstm' + str(self.layers_num),  l)
            self.model.add_module("part0"+str(self.layers_num), Part0())
            print("增加lstm层, 输入维度：{}, 输出维度：{},".format(kw['size'][0], kw['size'][1]))
        elif layer == 'batchnorm1d':
            self.model.add_module('batchnorm1d'+str(self.layers_num),  torch.nn.BatchNorm1d(kw['size']))
            print("增加batchnorm1d层, 维度：{}".format(kw['size']))
        elif layer == 'batchnorm2d':
            self.model.add_module('batchnorm2d'+str(self.layers_num),  torch.nn.BatchNorm2d(kw['size']))
            print("增加batchnorm2d层, 维度：{}".format(kw['size']))
        
        elif layer == "squeeze":
            self.model.add_module("squeeze", Squeeze())
        elif layer == "unsqueeze":
            self.model.add_module("unsqueeze", Unsqueeze())
        elif layer == 'part0':
            self.model.add_module("part0"+str(self.layers_num), Part0())
        elif layer is not None and not isinstance(layer, str):
            print("增加自定义模块{}".format(layer))
            self.model.add_module(str(layer.__class__.__name__)+str(self.layers_num), layer)
        elif layer =='action_model':
            l = torch.nn.LSTM(kw['size'][0], kw['size'][1], batch_first=True, bidirectional=False)
            l = l.to(self.device)
            self.model.add_module('lstm' + str(self.layers_num),  l)
            self.model.add_module("part0"+str(self.layers_num), Part0())
            print("增加lstm层, 输入维度：{}, 输出维度：{},".format(kw['size'][0], kw['size'][1]))
            self.model.add_module("dim2"+str(self.layers_num), dim2())
            self.model.add_module('batchnorm1d'+str(self.layers_num),  torch.nn.BatchNorm1d(kw['size'][1]))
            print("增加batchnorm层, 维度：{}".format(kw['size'][1]))
        elif layer == 'res_block':
            if  'num_blocks' not in kw.keys():
                kw['num_blocks'] = 1

            self.model.add_module('res_stage'+str(self.layers_num),ResNet(BasicBlock, kw['num_blocks'],kw['size'],stride))
            print("增加BasicBlock(ResNet)层, 输入维度：{}, 输出维度：{}, 层数： {} ,步长：{}".format(kw['size'][0], kw['size'][1], kw['num_blocks'],stride))
        elif layer == 'res_bottleneck':
            if  'num_blocks' not in kw.keys():
                kw['num_blocks'] = 1

            self.model.add_module('res_stage'+str(self.layers_num),ResNet(Bottleneck, kw['num_blocks'],kw['size'],stride))
            print("增加Bottleneck(ResNet)层, 输入维度：{}, 输出维度：{}, 层数： {} ,步长：{}".format(kw['size'][0], kw['size'][1], kw['num_blocks'],stride))
        elif layer == 'diffusion_model':
            self.diffusion_model = True
            from .blocks import GaussianDiffusion, Unet
            channels = 1 if self.color=='grayscale' else 3
            model = Unet(
                dim = 64,
                channels = channels,
                dim_mults = (1, 2, 2)
            )
            #定义扩散模型GaussianDiffusion类
            timestep = 500 if 'timestep' not in kw.keys() else kw['timestep']
            img_size = 28 if 'img_size' not in kw.keys() else kw['img_size']
            self.model = GaussianDiffusion(
                model,
                objective = 'pred_noise',
                image_size = img_size,
                timesteps = timestep,     # number of steps
            )
            # self.model.add_module('diffusion_model'+str(self.layers_num), diffusion)
            print("增加扩散模型层, 时间步数：{}。".format(timestep))
        elif layer == 'mobilenet':
            from .blocks import MobileNetV2
            # self.model = MobileNetV2(num_classes=1000)
            num_classes = self.num_classes
            self.model.add_module('mobilenetv2' + str(self.layers_num), MobileNetV2(num_classes=num_classes))
            print("增加MobileNetV2层。")
        elif layer == 'mobilenet_backbone':
            from .blocks import MobileNetV2_backbone
            # self.model = MobileNetV2(num_classes=1000)
            num_classes = self.num_classes
            self.model.add_module('mobilenetv2_backbone' + str(self.layers_num), MobileNetV2_backbone(num_classes=num_classes))
            print("增加MobileNetV2_backbone层。")

        else:
            islayer[0] = False

        if activation is not None:
            activation = activation.lower()
            # 激活函数
            if activation == 'relu':
                self.model.add_module('relu' + str(self.layers_num), torch.nn.ReLU())
                print("使用relu激活函数。")
            # elif activation == 'softmax':
            elif activation == 'tanh':
                self.model.add_module('tanh' + str(self.layers_num), torch.nn.Tanh())
                print('使用tanh激活函数。')
            elif activation == 'leakyrelu':
                self.model.add_module('leakyrelu' + str(self.layers_num), torch.nn.LeakyReLU(0.2))
                print('使用leakyrelu激活函数。')
            elif activation == 'sigmoid':
                self.model.add_module('sigmoid' + str(self.layers_num), torch.nn.Sigmoid())
                print('使用sigmoid激活函数。')

            else:
                self.model.add_module('softmax'+str(self.layers_num), torch.nn.Softmax())
                print('使用softmax激活函数。')
        else:
            islayer[1] = False

        # 优化器
        if optimizer != None:
            self.optimizer = optimizer
        else:
            islayer[2] = False
        
        if not any(islayer):
            print("No such Layer '{}'！".format(layer))

    def visual_feature(self, data, in1img=False, save_fold="layers"):
        if len(data.shape) == 1:
            data = np.reshape(data, (1,data.shape[0]))
            data = Variable(torch.tensor(np.array(data)).to(torch.float32))
            self.model.eval()
            f = open(os.path.join(save_fold,'layer_data.txt'), "w")
            str_data = ""
            act_layer = 0
            tra_layer = 0
            for num, i in enumerate(self.model):
                data = data.to(self.device)
                data = i(data)
                # print(num, i, data)
                if isinstance(i, (type(torch.nn.ReLU()),type(torch.nn.Softmax()))): # 非传统层，不计数
                    act_layer+=0.1
                    str_data += str(tra_layer-1+act_layer) + "."+str(i) +"\n" + str(np.squeeze(data).tolist()) + "\n"
                else:  #传统网络层
                    act_layer =0
                    str_data += str(tra_layer) + "."+str(i) +"\n" + str(np.squeeze(data).tolist()) + "\n"
                    tra_layer+= 1


            f.write(str_data)
            f.close()
        else:
            if len(data.shape) == 2:
                h,w = data.shape
                c = 1
            elif  len(data.shape) == 3:
                h,w,c = data.shape
            data = np.reshape(data, (1,c,h,w))
            data = Variable(torch.tensor(np.array(data)).to(torch.float32))
            self.model.eval()
            dir_name = os.path.abspath(save_fold)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            if not in1img: # 每层分别画一张图，横向
                for num, i in enumerate(self.model):
                    data = data.to(self.device)
                    data = i(data)
                    self.show_one_layer(data, i, num+1, dir_name)
            else: # 所有层共画一张图，纵向
                # fig, ax = plt.subplots(20, 20)
                plt.figure(figsize=(18,10))
                num_img = 0
                for name, para in self.model.named_parameters():
                    if "Linear" not in name:
                        num_img = max(num_img, para.size()[0])
                grid = plt.GridSpec(num_img+1,len(self.model)+1,wspace=0.5, hspace=0.5) #（ 单层最大图片数+1，层数）
                tra_layer = 1
                str_l = ""
                act_layer = 0
                for num, i in enumerate(self.model):
                    data = data.to(self.device)
                    data = i(data)
                    tmp_data = data
                    # print(i,"data.shape", data.shape)
                    if len(data.shape) > 2 and data.shape[0] == 1:
                        tmp_data = np.squeeze(data)

                    for j in range(tmp_data.shape[0]): # 每一层
                        # print("num+1",num+1, "j", j)
                        img = tmp_data[j].cpu().detach().numpy()
                        if len(img.shape) == 1:
                            img = np.reshape(img, (img.shape[0],1))
                        # w, c = img.shape
                        # print(w, c)
                        # img = np.reshape(img, (w, c))
                        # plt.subplot(tmp_data.shape[0],num+1, j+1)
                        if tmp_data.shape[0] == 1:
                            # ax[:, num+1].imshow(img)
                            ax = plt.subplot(grid[1:, num+1])
                            ax.imshow(img)
                        else:
                            # ax[ j,num+1].imshow(img)
                            ax = plt.subplot(grid[j+1, num+1])
                            ax.imshow(img)
                        # plt.imshow(img)
                        plt.xticks([])
                        plt.yticks([])
                    # print(num, i)
                    ax = plt.subplot(grid[0, num+1])
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                    if isinstance(i, (type(torch.nn.ReLU()),type(Reshape()),type(torch.nn.Softmax()))):
                        act_layer += 0.1
                        ax.text(0.5,0,tra_layer-1+act_layer , ha='center', va='center')
                        str_l += str(tra_layer-1+act_layer)+": "+str(i) + '\n'
                    else: # 传统网络层
                        act_layer = 0
                        ax.text(0.5,0,tra_layer, ha='center', va='center')
                        str_l += str(tra_layer)+": "+str(i) + '\n'
                        tra_layer +=1
                    # print(act_layer)
                ax = plt.subplot(grid[-1, 0])
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                plt.xticks([])
                plt.yticks([])
                # for i, layer in enumerate(self.model):
                #     str_l += str(i+1)+": "+str(layer) + '\n'
                ax.text(0,0,str_l)

                plt.savefig("{}/{}.jpg".format(dir_name, "total"))

                plt.show()

            print("Visualization result has been saved in {} !".format(dir_name))

    def extract_feature(self,data=None, pretrain=None):
        if len(data.shape) == 2:
            h,w = data.shape
            c = 1
        elif  len(data.shape) == 3:
            h,w,c = data.shape
        data = np.reshape(data, (1,c,h,w))
        data = Variable(torch.tensor(np.array(data)).to(torch.float32))
        if pretrain == None:
            self.model.eval()
            out = self.model(data)
        # elif pretrain == 'resnet34':
        else:
            from torchvision import models,transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(512),
                transforms.Resize(224),
                transforms.ToTensor()
            ])
            if len(data.shape) > 2 and data.shape[0] == 1:
                data = np.squeeze(data)
            data = transform(data)
            c,h,w = data.shape
            data = np.reshape(data, (1, c, h,w))
            # model = models.resnet34(pretrained=True)
            str = "models.{}(pretrained=True)".format(pretrain)
            model = eval(str)
            model.classifier = torch.nn.Sequential()
            model.eval()
            with torch.no_grad():
                out = model(data)
        out = out.detach().numpy()
        return out

    def show_one_layer(self, data, layer_name, num, dir_name):
        if len(data.shape) > 2 and data.shape[0] == 1:
            data = np.squeeze(data)

        for i in range(data.shape[0]):
            img = data[i].cpu().detach().numpy()
            if len(img.shape) == 1:
                img = np.reshape(img, (1, img.shape[0]))
            # w, c = img.shape
            # print(w, c)
            # img = np.reshape(img, (w, c))
            plt.subplot(1,data.shape[0], i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        plt.suptitle(layer_name)
        plt.savefig("{}/{}.jpg".format(dir_name, num))
        # plt.show()

    def dict2list(self, dict_data):
        if isinstance(dict_data, dict):
            if isinstance(list(dict_data.keys())[0],str): # 字典的键为类别，即classes2idx
                list_data = list(dict_data.keys())
            else: # 字典的值为类别，即idx2classes
                list_data = list(dict_data.values())
        else:
            list_data = dict_data
        return list_data


    def load_img_data(self, data_path, classes=None, batch_size=32,train_val_ratio=1.0, shuffle=True,batch_mode=False,transform=None,color="RGB",num_workers=0,**kw ):
        from .load_data import ImageFolderDataset
        from torchvision.transforms import transforms
        # 解析transform字典，通过compose获得transform对象
        if isinstance(transform, dict):
            tmp = []
            for k,v in transform.items():
                tran = "transforms.{}({})".format(k,v)
                tran = eval(tran)
                tmp.append(tran)
            tmp = transforms.Compose(tmp)
            self.transform = tmp
        else:
            self.transform = transform

        self.dataset_type = "img"
        dataset = ImageFolderDataset(data_path, transform=self.transform,color=color, batch_mode=batch_mode,scale=self.scale)
        self.dataset_size = int(len(dataset))
        self.color = color
        self.img_classes = self.dict2list(classes )
        
        self.num_classes = len(set(dataset.labels)) if self.img_classes is None else len(self.img_classes)
        if 0 < train_val_ratio < 1:
            train_size =  int(train_val_ratio * self.dataset_size)
            val_size =  self.dataset_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1

            return self.dataloader, self.val_dataloader

        else:
            self.dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.label = torch.Tensor()
            for train_x, train_y in self.dataloader:
                # test_x = test_x.to(self.device)
                # test_y = test_y.to(self.device)
                # batch_res = self.model(test_x)
                self.label = torch.cat([self.label, train_y], dim=0)
            self.label = self.label.cpu().detach().numpy()
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1
            return self.dataloader

    def load_tab_data(self, data_path,batch_size=32, train_val_ratio=1.0, shuffle=True,y_type='long',num_workers=0,**kw):
        from .load_data import TabularDataset
        self.dataset_type = "tab"
        if y_type == 'long' and self.task_type == 'reg':
            y_type = 'float'
        dataset = TabularDataset(data_path,y_type=y_type,task_type=self.task_type)
        self.dataset_size = int(len(dataset))
        # print(self.dataset_size)
        # self.dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle,num_workers=1)
        if 0 < train_val_ratio < 1:
            train_size =  int(train_val_ratio * self.dataset_size)
            val_size =  self.dataset_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1
            return self.dataloader, self.val_dataloader
        else:
            self.dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1
            return self.dataloader

    def load_npz_data(self, data_path, batch_size=32, shuffle=True, classes=None,train_val_ratio=1.0,num_workers=0,**kw):
        from .load_data import NpzDataset
        self.dataset_type = "npz"

        dataset = NpzDataset(data_path)
        self.dataset_size = int(len(dataset))
        # print(self.dataset_size)
        self.img_classes = classes
        try:
            self.word2idx = dataset.word2idx
        except:
            pass
        # print(self.dataset_size)
        # self.dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle,num_workers=1)
        if 0 < train_val_ratio < 1:
            train_size =  int(train_val_ratio * self.dataset_size)
            val_size =  self.dataset_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1
        else:
            self.dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            self.input_size = np.array(next(iter(self.dataloader))[0].shape)
            self.input_size[0] = 1

        return self.dataloader

    def load_dataset(self, x, y,num_workers=0, **kw):
        self.dataset_type = 'other'
        if kw:
            if 'word2idx'  in kw:
                print("载入词表")
                self.word2idx = kw['word2idx']
                self.idx2word = {v:k for k, v in self.word2idx.items()}
                self.x = torch.from_numpy(x)
                self.y = torch.from_numpy(y)
            if 'classes' in kw:
                # print(kw['classes'])
                if isinstance(kw['classes'],dict):
                    if isinstance(list(kw['classes'].keys())[0],str): # 字典的键为类别，即classes2idx
                        self.img_classes = list(kw['classes'].keys())
                    else: # 字典的值为类别，即idx2classes
                        self.img_classes = list(kw['classes'].values())

                else:
                    self.img_classes = kw['classes']

                self.x = Variable(torch.tensor(np.array(x)).to(torch.float32))
                self.y = Variable(torch.tensor(np.array(y)).long())
        else:
            # self.x = Variable(torch.tensor(np.array(x)).to(torch.float32))
            self.x = torch.from_numpy(x).to(torch.float32)
            # self.y = Variable(torch.tensor(np.array(y)).long())
            self.y = torch.from_numpy(y)

            self.batchsize = self.x.shape[0]
        
        import torch.utils.data as Data
        from torch.utils.data import DataLoader
        dataset = Data.TensorDataset(self.x, self.y)
        self.dataset_size = int(len(dataset))

        self.dataloader = Data.DataLoader(dataset,
                    batch_size = len(dataset),
                    shuffle = True,
                    num_workers = num_workers)

    def set_seed(self, seed):# 设置随机数种子
        import random
        torch.manual_seed(seed)   #CPU
        torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子（只用一块GPU）
        torch.cuda.manual_seed_all(seed) # 所有GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

    def train(self, lr=0.1, epochs=30, batch_num=1, batch_size=16, save_fold=None, loss="CrossEntropyLoss" ,metrics=[],filename="basenn.pth", checkpoint=None):
        if self.task_type == 'reg' and 'acc' in metrics:
            metrics.remove('acc')
        if  self.task_type == 'reg' and 'mse' not in metrics:
            metrics.append('mse')
        if self.rnn: # 针对文本任务
            import torch.utils.data as Data
            # from torch.utils.data import DataLoader
            # dataset = Data.TensorDataset(self.x, self.y)
            # self.dataloader = Data.DataLoader(dataset,
            #             batch_size = batch_size,
            #             shuffle = True,
            #             num_workers = 1)
            if checkpoint and self.device== torch.device('cpu'):
                self.model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')))
            elif checkpoint:
                self.model.load_state_dict(torch.load(checkpoint)['state_dict'])
                        
            self.model.to(self.device)

            # 设置优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
            # scheduler = torch.optim.lr_sheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
            # 设置损失函数
            criterion = torch.nn.CrossEntropyLoss()

            # 定义训练过程
            for epoch in range(epochs):
                for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader,0):

                    batch_x = batch_x.long().transpose(1, 0).contiguous()
                    batch_y = batch_y.long().transpose(1, 0).contiguous()
                    # print(batch_x.shape, batch_y.shape)
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    output, _ = self.model(batch_x)
                    loss = criterion(output, batch_y.view(-1))

                    # if batch_idx % 900 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, batch_idx * len(batch_x[1]), len(self.dataloader.dataset),
                            100. * batch_idx / len(self.dataloader), loss.item()))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

            # 保存模型
            print("保存模型中...")

            info = {
                'meta':{
                    'tool':'BaseNN', 
                    'backbone':self.model, 
                    'device':self.device,
                    'dataset_size':self.dataset_size,
                    'learning_rate':lr,
                    'epoch':epochs,
                    'time': time.asctime( time.localtime(time.time()) ),
                    'word2idx': self.word2idx,
                    'input_size':self.input_size,
                    'dataset_type':self.dataset_type,
                },
                'state_dict':self.model.state_dict(),
                'para':{
                    'config':self.config,
                    'rnn':self.rnn,
                }

            }
            if save_fold is not None:
                self.save_fold = save_fold
            if self.save_fold is not None:
                if not os.path.exists(self.save_fold):
                    os.mkdir(self.save_fold)
                model_path = os.path.join(self.save_fold, filename)
                torch.save(info, model_path)
                print("保存模型{}成功！".format(model_path))
            else:
                torch.save(info, filename)
                print("保存模型{}成功！".format(filename))

        else: # 针对图像任务
            if checkpoint:
                if not os.path.exists(checkpoint):
                    print("未找到{}文件！".format(checkpoint))
                    return 
                self.model = torch.load(checkpoint,map_location=torch.device('cpu'))['state_dict']

            self.loader = self.dataloader
            loss_str = "torch.nn.{}()".format(loss)
            loss_fun = eval(loss_str)
            # print("损失函数：", loss_fun)

            if self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9)  # 使用SGD优化器
            elif self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif self.optimizer == 'Adagrad':
                optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr)
            elif self.optimizer == 'ASGD':
                optimizer = torch.optim.ASGD(self.model.parameters(), lr=lr)  
            print("使用 {} 优化器。".format(self.optimizer))
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
            self.model.to(self.device)
            log = []
            for epoch in range(epochs):  
                # b_loss = 0
                # b_acc = 0
                for batch_x, batch_y in self.loader:
                    # a = time.time()
                    # print("1", a)
                    # print("iter ", iter, np.squeeze(batch_x[0]).shape, batch_y[0])
                    # import cv2
                    # cv2.imshow("asd", np.array(np.squeeze(batch_x[0])))
                    # print("batch_y[0]",np.squeeze(batch_y[0]))
                    # y_pred = self.model(self.x)
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()  # 将梯度初始化为零，即清空过往梯度
                    # print(batch_x.shape,type(batch_x))
                    y_pred = self.model(batch_x)
                    if self.task_type == 'cls' and (len(batch_y.shape) == 1 or batch_y.shape[-1] == 1):
                        from torch.nn.functional import one_hot
                        batch_y = one_hot(batch_y, self.last_channel).float()
                    # print(batch_y[0],y_pred[0])
                    # cv2.waitKey(0)
                    if self.diffusion_model:
                        loss = y_pred
                    else:
                        loss = loss_fun(y_pred, batch_y)
                    loss.backward()  # 反向传播，计算当前梯度
                    optimizer.step()  # 根据梯度更新网络参数\
                    # scheduler.step()
                    # print("2", a - time.time())

                    log_info = "{epoch:%d  Loss:%.4f}" % (epoch, loss)
                    log_dict = {"epoch":epoch, "loss":loss.item()}
                    if "acc" in metrics:
                        acc = cal_accuracy(batch_y, y_pred)
                        log_info = log_info[:-1] # 去掉句末大括号
                        log_info+= "  Accuracy:%.4f}"%acc # 加入acc信息
                        log_dict['acc'] = acc
                    if  "mae" in metrics:
                        mae = torch.nn.L1Loss()
                        mae = mae(y_pred, batch_y)
                        log_info = log_info[:-1] # 去掉句末大括号
                        log_info+= "  MAE:%.4f}"%mae # 加入acc信息
                        log_dict['mae'] = mae
                    if "mse" in metrics:
                        mse = torch.nn.MSELoss()
                        mse = mse(y_pred, batch_y)
                        log_info = log_info[:-1] # 去掉句末大括号
                        log_info+= "  MSE:%.4f}"%mse # 加入acc信息
                        log_dict['mse'] = mse
                    print(log_info)
                    log.append(log_dict)
                    # b_acc += acc
                    # b_loss+= loss

                # step+=1
                # print("{epoch:%d  Loss:%.4f  Accuracy:%.4f}" % (epoch, b_loss/step, b_acc/step),step)

            if save_fold:
                self.save_fold = save_fold
                # print(self.save_fold)
            if not os.path.exists(self.save_fold):
                os.mkdir(self.save_fold)

            model_path = os.path.join(self.save_fold, filename)
            print("保存模型中...")
            info = {
                'meta':{
                    'tool':'BaseNN', 
                    'backbone':self.model, 
                    'device':self.device,
                    'dataset_size':self.dataset_size,
                    'learning_rate':lr,
                    'epoch':epochs,
                    'time': time.asctime( time.localtime(time.time()) ),
                    'color':self.color,
                    'transform':self.transform, 
                    'input_size':self.input_size,
                    'dataset_type':self.dataset_type,
                },
                'state_dict':self.model,
                'para':{
                    'rnn':self.rnn,
                    'diffusion':self.diffusion_model,
                }

            }
            try:
                info['meta']['CLASSES'] = self.img_classes
            except:
                pass
            torch.save(info,model_path)
            # torch.save([self.img_classes, self.model], model_path)
            print("保存模型{}成功！".format(model_path))
            return log

    @torch.no_grad()
    def inference(self, data=None, show=False, checkpoint=None,hidden=None, label=False,num=4,return_all_timesteps=True):
        if checkpoint is not None:
            ckpt_para = torch.load(checkpoint,map_location=torch.device('cpu'))['para']
            ckpt_meta = torch.load(checkpoint,map_location=torch.device('cpu'))['meta']
            self.rnn = ckpt_para['rnn']
            self.model = torch.load(checkpoint,map_location=torch.device('cpu'))['state_dict']
            if 'diffusion' in ckpt_para:
                self.diffusion_model = ckpt_para['diffusion']
            
            color = ckpt_meta['color']
            if 'transform' in ckpt_meta:
                transform = ckpt_meta['transform']
            else:
                transform = self.transform
            if 'CLASSES' in ckpt_meta:
                self.img_classes = ckpt_meta['CLASSES']
            self.model.to(self.device)

        else:
            color = self.color
            transform = self.transform
        if self.rnn:
            self.word2idx = torch.load(checkpoint)['meta']['word2idx']
            self.ix2word = {v:k for k, v in self.word2idx.items()}
            config = torch.load(checkpoint)['para']['config']
            self.model =  lstm(len(self.word2idx),config['para']['ed'],config['para']['hd'],config['para']['nl'])
            self.model.eval()
            self.model.to(self.device)
            input = torch.Tensor([self.word2idx[data]]).view(1, 1).long()
            output, hidden = self._inference(data=input,checkpoint=checkpoint,hidden=hidden)
            output = output.data[0].cpu()
            lay = torch.nn.Softmax(dim=0)
            output = lay(output)
            return np.array(output),hidden
        elif self.diffusion_model:
            self.model.to(self.device)
            generated_images = self.model.sample(batch_size=num, return_all_timesteps=return_all_timesteps)
            return generated_images.cpu().detach().numpy()

        else:
            if isinstance(data,str): # 推理文件路径
                file_type = data.split(".")[-1]
                if file_type == "csv":  # 推理csv
                    if not label: # 无标签
                        x = np.loadtxt(data, dtype=float, delimiter=',',skiprows=1)
                        data  = Variable(torch.tensor(np.array(x)).to(torch.float32)).to(self.device)
                    else: # 有标签
                        x = np.loadtxt(data, dtype=float, delimiter=',',skiprows=1)
                        data = x[:, :-1]
                        data  = Variable(torch.tensor(np.array(data)).to(torch.float32)).to(self.device)
                elif file_type.lower() in ["jpg", "png", "jpeg"]:  # 推理图片
                    # img = cv2.imread(data)
                    if color == "grayscale":
                        img = cv2.imread(data,cv2.IMREAD_GRAYSCALE) /self.scale
                        img = np.expand_dims(img, 2)                 
                    else:
                        img = cv2.imread(data) / self.scale
                    if transform is not None:
                        # img = transform(img)
                        img = np.array(transform(Image.fromarray((img*self.scale).astype(np.uint8)))) /self.scale
                    data = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(torch.float32).to(self.device) 
                    # if len(img.shape) == 2:
                    #     # 灰度，加两维度（sample，channel）
                    #     data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device) 
                    # elif len(img.shape) == 3:
                    #     # 灰度/RGB，加一维度（sample）
                    #     data = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(torch.float32).to(self.device) 
                    # else:
                    #     print("传入图像的形状不符合要求：",img.shape)
                    #     return
                elif os.path.isdir(data): # 推理文件夹
                    img_list = []
                    dir_list = os.listdir(data)
                    # print(dir_list)
                    # 将顺序读取的文件保存到该list中
                    for item in dir_list:
                        img_path = os.path.join(data,item)
                        # img = np.array(Image.open(img_path))
                        color = torch.load(checkpoint,map_location=torch.device('cpu'))['meta']['color']
                        if color == "grayscale":
                            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                            img = np.expand_dims(img, 2)                 

                        else:
                            img = cv2.imread(img_path)
                        # if len(img.shape) == 2:
                        #     # 灰度，加1维度（channel）
                        #     img = np.expand_dims(img, 0)                 
                        transform = torch.load(checkpoint,map_location=torch.device('cpu'))['meta']['transform']
                        if transform is not None:
                            img = np.array(transform(Image.fromarray(img)))
                        img_list.append(img)
                    
                    data = torch.from_numpy(np.array(img_list)).permute(0,3,1,2).to(torch.float32).to(self.device) 
                    # x = np.expand_dims(x, axis=1)
            elif isinstance(data, DataLoader):  # 推理dataloader
                res = torch.Tensor().to(self.device)
                self.val_label = torch.Tensor().to(self.device)
                for test_x, test_y in data:
                    test_x = test_x.to(self.device)
                    test_y = test_y.to(self.device)
                    batch_res = self.model(test_x)
                    res = torch.cat([res, batch_res], dim=0)
                    self.val_label = torch.cat([self.val_label, test_y], dim=0)
                self.val_label = self.val_label.cpu().detach().numpy()
                res = res.cpu().detach().numpy()
                if show:
                    print("推理结果为：",res)
                self. res = res
                return res

            else:  # 推理numpy数组
                transform = torch.load(checkpoint,map_location=torch.device('cpu'))['meta']['transform']
                if transform is not None:
                    data = np.array(transform(Image.fromarray(data)))
                # if len(np.array(data).shape) == 3:
                #     data = torch.from_numpy(data).unsqueeze(0).permute(0,3,1,2).to(torch.float32).to(self.device) 
                # else:
                data  = Variable(torch.tensor(np.array(data)).to(torch.float32)).to(self.device)

            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                res = self.model(data)
            res = res.cpu().detach().numpy()
            if show:
                print("推理结果为：",res)
            self. res = res
            return res
    
    def _inference(self, data, show=False, checkpoint=None,hidden=None):
        data = data.to(self.device)

        if checkpoint and self.device== torch.device('cpu'):
            self.model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu'))[1])
        elif checkpoint:
            sta = torch.load(checkpoint,map_location=torch.device('cpu'))['state_dict']
            self.model.load_state_dict(sta)

        self.model.to(self.device)
        output, hidden = self.model(data, hidden)
        return output, hidden
    
    def noisy(self, data, timestep=500,num=10,show=True):
        """正向加噪过程
        Args:
            data (str|numpy.ndarray): 输入图像.
            timestep (int, optional): 时间步数. Default to 500.
            num (int, optional): 生成图像数量. Default to 10.
            show (bool, optional): 是否显示生成图像. Default to True.

        Returns:
            result: 生成的图像列表
        """
        if isinstance(data,str):
            image = Image.open(data)
        else:
            image = data
        from torchvision import transforms
        Totensor = transforms.ToTensor()
        x_start = Totensor(image)
        # 展示num张扩散过程
        result = []
        for idx in range(num):
        # for idx, t in enumerate([0, 50, 100, 150, 200, 300, 400, 499]):
            t = timestep * idx // num
            x_noisy = self.model.q_sample(x_start, t=torch.tensor([t])) #在GPU算力下运行
            x_noisy = x_noisy.numpy()
            result.append(list(x_noisy))
            x_noisy = np.clip(x_noisy, 0, 1)
            if self.color == "grayscale":
                x_noisy = np.squeeze(x_noisy)
            else:
                x_noisy = np.squeeze(np.transpose(x_noisy, (1, 2, 0)))
            # print(x_noisy.shape)
            plt.subplot(1, num, 1 + idx) 
            if self.color == "grayscale":
                plt.imshow(x_noisy, cmap="gray")
            else:
                plt.imshow(x_noisy)
            plt.axis("off")
            # plt.title(f"t={t}")
        if show:
            plt.show()
        return result

    def show(self, data, size=(2,2), visual_timesteps=False):
        timesteps = data.shape[1]
        num = data.shape[0]
        fig = plt.figure( constrained_layout=True)
        gs = fig.add_gridspec(size[0], size[1])
        if not visual_timesteps:
            assert size[0] * size[1] == num, "Error Code: -308. The number of generated images does not match the number of images to be displayed."
            #设定每张图像显示大小（16，16），设定生成显示有8行8列的图像

            #前两个数字8代表生成8行8列的图像，501代表timestep=500，中间的3代表3通道，后面两个28代表图像本身分辨率为28*28；
            if self.color == "grayscale":
                imgs = data.reshape(size[0], size[1], data.shape[1], data.shape[3], data.shape[4])
            else: 
                imgs = data.reshape(size[0], size[1], data.shape[1], data.shape[2],data.shape[3], data.shape[4])
                imgs = np.transpose(imgs, (0, 1,2, 4, 5, 3))

            for n_row in range(size[0]):
                for n_col in range(size[1]):
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    img = imgs[n_row, n_col, -1]
                    img = img.astype(np.float32)
                    Img = np.maximum(img, 0)
                    Img = np.minimum(Img, 1)
                    #img = (img.numpy() + 1) * 127.5
                    #img = img.astype(np.uint8)
                    if self.color == "grayscale":
                        f_ax.imshow(Img,cmap='gray')
                    else:
                        f_ax.imshow(Img)
                    f_ax.axis("off")  
            # 显示图像
                    

        else:
            assert size[0] <= num and size[1] <= timesteps, "Error Code: -308. The number of generated images does not match the number of images to be displayed."
            # imgs = data.reshape(64, 501, 3, 28, 28)
            if self.color == "RGB":
                data = np.transpose(data, (0, 1,3, 4, 2))
            data = data.astype(np.float32)
            data = np.clip(data, 0, 1)
            #外循环：在生成的64个图像中，显示其中前16张图像，所以看到结果有16个不同的手写数字；
            #内循环：从每张图像的0-500的timestep过程中，间隔选择16张图像显示其由噪声生成手写数字的过程；
            for n_row in range(size[0]):
                for n_col in range(size[1]): # 0-10
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    
                    t_idx = (timesteps // size[1]) * n_col if n_col < size[1]-1 else -1
                    if self.color == "grayscale":
                        f_ax.imshow((np.squeeze(data[n_row, t_idx])), cmap="gray")
                    else:
                        f_ax.imshow(data[n_row, t_idx])
                    f_ax.axis("off")
        
        plt.show()


    def print_model(self):
        # print('模型共{}层'.format(self.layers_num))
        print(self.model)

    def save(self, model_path='basenn.pth'):
        print("保存模型中...")
        torch.save(self.model, model_path)
        print("保存模型{}成功！".format(model_path))
    
    def load(self,model_path):
        print("载入模型中...")
        # self.model = torch.load(model_path)
        self.model = torch.load(model_path,map_location=torch.device('cpu'))['state_dict']

        self.model.to(self.device)
        print("载入模型{}成功！".format(model_path))

    def print_result(self, result=None):
        res_idx = self.res.argmax(axis=1)
        res = {}
        for i,idx in enumerate(res_idx):
            try:
                pred = self.img_classes[idx]
            except:
                pred = idx
            res[i] ={"预测值":pred,"置信度":self.res[i][idx]} 
        print("推理结果为：", res)
        return res
    
    def convert(self, checkpoint=None,out_file="convert_model.onnx",backend="ONNX", opset_version=10,ir_version=6):
        # if checkpoint is not None:
        ckpt = torch.load(checkpoint,map_location=torch.device('cpu'))
        model = ckpt['state_dict']
        dataset_type = ckpt['meta']['dataset_type']        
        input_size = ckpt['meta']['input_size']
        # print(input_size, dataset_type)
        x = torch.randn(tuple(input_size)) # tab iris 10,4 ;  image mn 1,3,128,128 ; 
        tang = False
        if dataset_type == "img":
            with torch.no_grad():
                torch.onnx.export(model, 
                    x,
                    out_file,
                    opset_version=opset_version,
                    # do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size",2:"width",3:"height"},  # 批处理变量
                                    "output":{0:"batch_size",2:"width",3:"height"}})
                print(f'Successfully exported ONNX model: {out_file}')

            with open(out_file.replace(".onnx", ".py"), "w+") as f:
                gen0 = """
from XEdu.hub import Workflow as wf
import numpy as np

# 模型声明
basenn = wf(task='basenn',checkpoint='{}')
# 待推理图像，此处仅以随机数组为例，以下为{}张{}通道的{}*{}的图像。
image = np.random.random({}).astype('{}') # 可替换成您想要推理的图像路径,如 image = 'cat.jpg'
# 模型推理
res = basenn.inference(data=image)
# 标准化推理结果
result = basenn.format_output(lang="zh")

# 更多用法教程详见：https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html
"""
                gen = gen0.format(out_file,  x.shape[0], x.shape[1],x.shape[2], x.shape[3],tuple(x.shape).__str__(),str(x.dtype).split('.')[-1],).strip('\n')
                # gen = gen0.strip('\n') + out_file + gen1.strip('\n') + tuple(input_size).__str__() +gen2.strip('\n')
                f.write(gen)

        elif dataset_type == "tab":
            with torch.no_grad():
                torch.onnx.export(model, 
                            x,
                            out_file,
                            opset_version=opset_version,
                            # do_constant_folding=True,	# 是否执行常量折叠优化
                            input_names=["input"],	# 输入名
                            output_names=["output"],	# 输出名
                            dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                            "output":{0:"batch_size"}})
                print(f'Successfully exported ONNX model: {out_file}')
            with open(out_file.replace(".onnx", ".py"), "w+") as f:
                gen0 = """
from XEdu.hub import Workflow as wf
import numpy as np

# 模型声明
basenn = wf(task='basenn',checkpoint='{}')
# 待推理数据，此处仅以随机二维数组为例，以下为{}个维度为{}的特征
table = np.random.random({}).astype('{}')
# 模型推理
res = basenn.inference(data=table)
# 标准化推理结果
result = basenn.format_output(lang="zh")

# 更多用法教程详见：https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html
"""
                feature_dim = np.squeeze(np.array(x.shape[1:]))
                gen = gen0.format(out_file, x.shape[0], feature_dim, tuple(x.shape).__str__(),str(x.dtype).split('.')[-1]).strip('\n')
                # gen = gen0.strip('\n') + out_file + gen1.strip('\n') + tuple(input_size).__str__() +gen2.strip('\n')
                f.write(gen)
        elif dataset_type == 'npz':
            model = ckpt['meta']['backbone']
            if ckpt['para']['rnn']:
                x = torch.randint(10,(1,1))         
                tang = True   
            with torch.no_grad():
                torch.onnx.export(model, # tab iris
                    x,
                    out_file,
                    opset_version=opset_version,
                    # do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})
                print(f'Successfully exported ONNX model: {out_file}')
            with open(out_file.replace(".onnx", ".py"), "w+") as f:
                gen0 = """
from XEdu.hub import Workflow as wf
import numpy as np

# 模型声明
basenn = wf(task='basenn',checkpoint='{}')
# 待推理数组，此处仅以随机数组为例
data = np.random.random({}).astype('{}') # 可替换成您想要推理的npz文件路径,如 data = 'action.npz',npz文件中至少应该包括一个键：data,其中存储数据信息（数组形式）。
# 模型推理
res = basenn.inference(data=data)
# 标准化推理结果
result = basenn.format_output(lang="zh")

# 更多用法教程详见：https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html
"""         
                gen = gen0.format(out_file, tuple(x.shape).__str__(), str(x.dtype).split('.')[-1]).strip('\n')
                # gen = gen0.strip('\n') + out_file + gen1.strip('\n') + tuple(input_size).__str__() +gen2.strip('\n')
                f.write(gen)
            
        # 插入信息
        from onnx import load_model,save_model
        onnx_model = load_model(out_file)
        onnx_model.ir_version = ir_version
        # model_info = {'codebase': 'BaseNN', 'input_size': input_size.tolist()}
        meta = onnx_model.metadata_props.add()
        meta.key = 'input_size'
        meta.value = str(input_size.tolist())
        meta = onnx_model.metadata_props.add()
        meta.key = 'dataset_type'
        meta.value = dataset_type
        if tang:
            meta = onnx_model.metadata_props.add()
            meta.key = 'word2idx'
            meta.value = str(ckpt['meta']['word2idx'])
            # print(meta.value)
        save_model(onnx_model, out_file)