
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from models.GSoP import *

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, ms_block=False, basewidth=26, scale=4, attention='0', att_dim=128, stype='normal'):
        super(Bottleneck, self).__init__()
        
        width = int(math.floor(planes * (basewidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        self.ms_block = ms_block
        if self.ms_block:
            if scale == 1:
                self.nums = 1
            else:
                self.nums = scale -1
            if stype == 'stage':
                self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
            convs = []
            bns = []
            for i in range(self.nums):
                convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
                bns.append(nn.BatchNorm2d(width))
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

            self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stype = stype
            self.scale = scale
            self.width  = width
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)

        self.dimDR = att_dim
        self.relu_normal = nn.ReLU(inplace=False)
        
        if attention in {'1','+','M','&'}:
            if planes > 64:
                DR_stride=1
            else:
                DR_stride=2

            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(
                 planes * self.expansion, self.ch_dim, 
                 kernel_size=1,stride=DR_stride, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            #row-wise conv is realized by group conv
            self.row_conv_group = nn.Conv2d(
                 self.ch_dim, 4*self.ch_dim, 
                 kernel_size=(self.ch_dim, 1), 
                 groups = self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(
                 4*self.ch_dim, planes * self.expansion, 
                 kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
        
        if attention in {'2','+','M','&'}:
            self.sp_d = att_dim
            self.sp_h = 8
            self.sp_w = 8
            self.sp_reso = self.sp_h * self.sp_w
            self.conv_for_DR_spatial = nn.Conv2d(
                 planes * self.expansion, self.sp_d, 
                 kernel_size=1,stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h,self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            #row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d( 
                 self.sp_reso, self.sp_reso*4, kernel_size=(self.sp_reso, 1), 
                 groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Conv2d(
                 self.sp_reso*4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

        if attention == '&':#we employ a weighted spatial concat to keep dim
            self.groups_base = 32
            self.groups = int(planes * self.expansion / 64)
            self.factor = int(math.log(self.groups_base / self.groups, 2))
            self.padding_num = self.factor + 2
            self.conv_kernel_size = self.factor * 2 + 5
            self.dilate_conv_for_concat1 = nn.Conv2d(planes * self.expansion, 
                                                    planes * self.expansion, 
                                                    kernel_size=(self.conv_kernel_size,1), 
                                                    stride=1, padding=(self.padding_num,0),
                                                    groups=self.groups, bias=True)
            self.dilate_conv_for_concat2 = nn.Conv2d(planes * self.expansion, 
                                                    planes * self.expansion, 
                                                    kernel_size=(self.conv_kernel_size,1), 
                                                    stride=1, padding=(self.padding_num,0),
                                                    groups=self.groups, bias=True)
            self.bn_for_concat = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.attention = attention

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out) # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous() # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out) # Nx512x1x1

        out = self.fc_adapt_channels(out) #NxCx1x1
        out = self.sigmoid(out) #NxCx1x1

        return out


    def pos_att(self, out):
        pre_att = out # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)

        out = self.adppool(out) # keep the feature map size to 8x8

        out = cov_feature(out) # Nx64x64
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out) # Nx256x1x1
        out = self.relu(out)

        out = self.fc_adapt_channels_for_spatial(out) #Nx64x1x1
        out = self.sigmoid(out) 
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()#Nx1x8x8

        out = self.adpunpool(out,(pre_att.size(2), pre_att.size(3))) # unpool Nx1xHxW

        return out


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.ms_block:
            spx = torch.split(out, self.width, 1)
            for i in range(self.nums):
                if i==0 or self.stype=='stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                if i==0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
            if self.scale != 1 and self.stype=='normal':
                out = torch.cat((out, spx[self.nums]),1)
            elif self.scale != 1 and self.stype=='stage':
                out = torch.cat((out, self.pool(spx[self.nums])),1)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.attention == '1': #channel attention,GSoP default mode
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        elif self.attention == '2': #position attention
            pre_att = out
            att = self.pos_att(out)
            out = self.relu_normal(pre_att * att)

        elif self.attention == '+': #fusion manner: average
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = pre_att * chan_att + self.relu(pre_att.clone() * pos_att)

        elif self.attention == 'M': #fusion manner: MAX
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = torch.max(pre_att * chan_att, self.relu(pre_att.clone() * pos_att))

        elif self.attention == '&': #fusion manner: concat
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out1 = self.dilate_conv_for_concat1(pre_att * chan_att)
            out2 = self.dilate_conv_for_concat2(self.relu(pre_att * pos_att))
            out = out1 + out2
            out = self.bn_for_concat(out)
        
        out += residual
        out = self.relu(out)

        return out

class UsqNet(nn.Module):
    def __init__(self, block, layers, ms_block=False, lsop_block=False, gsop_block=False, num_classes=5):
        super(UsqNet, self).__init__()
        
        self.ms_block = ms_block
        self.lsop_block = lsop_block
        self.gsop_block = gsop_block

        self.inplanes = 64
        self.baseWidth = 26
        self.scale = 4
        
        att_dim = 128
        att_mode = '1' if lsop_block else '0'
        att_position = [['0']*2+[att_mode],['0']*3+[att_mode],['0']*5+[att_mode],['0']*2+[att_mode]]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], att_position=att_position[0], att_dim=att_dim)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_position=att_position[1], att_dim=att_dim)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_position=att_position[2], att_dim=att_dim)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_position=att_position[3], att_dim=att_dim)

        if not self.gsop_block:
            self.avgpool = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d(14, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #print("Global average pooling...")

        else: 
            self.isqrt_dim = 256
            self.layer_reduce = nn.Conv2d(512 * block.expansion, self.isqrt_dim, kernel_size=1, stride=1, padding=0,
                                          bias=False)
            self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
            self.layer_reduce_relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1) / 2), num_classes)
            #print("local second-order pooling...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, att_position=[1], att_dim=128):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.ms_block, self.baseWidth, \
                            self.scale, att_position[0], att_dim=att_dim, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, \
                            ms_block=self.ms_block, basewidth=self.baseWidth, scale=self.scale, \
                            attention=att_position[i], att_dim=att_dim))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        x4 = x

        if not self.gsop_block: # Global average pooling
            x = self.avgpool(x)

        else: # Global Second-order Pooling (GSoP)
            x = self.layer_reduce(x)
            x = self.layer_reduce_bn(x)
            x = self.layer_reduce_relu(x)

            x = CovpoolLayer(x)
            x = SqrtmLayer(x, 3)
            x = TriuvecLayer(x)

        x = x.view(x.size(0), -1)
        x5 = x
        x = self.fc(x)

        return x, x1, x2, x3, x4, x5

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))


def USQNET(ms_block = True, lsop_block = True, gsop_block = True, num_classes=5, **kwargs):
    """Constructs a USQNet model.
    Args:
        ms_block (bool): True for including multi-scale block in bottleneck layer of resnet50
        lsop_block (bool): True for including local second-rder pooling block in bottleneck layer of resnet50
        gsop_block (bool): True for including global second-rder pooling block in place of global avergae pooling
    """
    if ms_block and lsop_block and gsop_block:
        print('Model: USQNet')
    if not ms_block and not lsop_block and not gsop_block:
        print('Model: RN50+GAP')
    if ms_block and not lsop_block and not gsop_block:
        print('Model: RN50+MS+GAP')
    if ms_block and lsop_block and not gsop_block:
        print('Model: RN50+MS+LSoP+GAP')
    if not ms_block and lsop_block and gsop_block:
        print('Model: RN50+L2GSoP')

    model = UsqNet(Bottleneck, [3, 4, 6, 3], ms_block = ms_block, lsop_block = lsop_block, gsop_block = gsop_block, **kwargs)
    num_ftrs = model.fc.in_features
    if ms_block or lsop_block or gsop_block:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                    nn.ReLU(inplace=True), 
                                    nn.Dropout(0.4),
                                    nn.Linear(256, num_classes))
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    net = USQNET() #USQNet
    # net = USQNET(ms_block=False, lsop_block=False, gsop_block=False) #RN50+GAP
    # net = USQNET(ms_block=True, lsop_block=False, gsop_block=False) #RN50+MS+GAP
    # net = USQNET(ms_block=True, lsop_block=True, gsop_block=False) #RN50+MS+LSoP+GAP
    # net = USQNET(ms_block=False, lsop_block=True, gsop_block=True) #RN50+L2GSoP
    net = net.cuda(0)
    # print(model)
    total_params = sum(p.numel() for p in net.parameters())
    total_train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total Param: ", total_params)
    print("Total Train Param: ", total_train_params)
    out = net(images)
    print(out[0].shape)