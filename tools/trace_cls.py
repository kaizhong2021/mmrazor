# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmcls.structures import ClsDataSample
from mmengine.config import Config, DictAction

from mmrazor.registry import MODELS
from mmrazor.utils import register_all_modules
# from mmrazor.fx import (CustomTracer,
#                                   UntracedMethodRegistry,
#                                   register_skipped_method,
#                                   custom_trace)
from torch.quantization.quantize_fx import _swap_ff_with_fxff

import torch.fx
from torch.fx import GraphModule
from torch.ao.quantization.fx import fuse


def parse_args():
    parser = argparse.ArgumentParser(description='Train an algorithm')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def is_same_node(node1, node2):
    if node1.name == node2.name and node1.target == node2.target and node1.prev == node2.prev:
        return True
    return False

def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build model
    model = MODELS.build(cfg.model)
    _swap_ff_with_fxff(model)

    print(model.graph_loss)
    dummy_input = torch.rand(1, 3, 224, 224)
    data_samples = [ClsDataSample().set_gt_label(1)]

    model.prepare('loss')
    loss = model.graph_loss(dummy_input, data_samples)
    print(loss)
    # model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # tracer = CustomTracer(customed_skipped_method=model.customed_skipped_method)
    # model.mode='predict'
    # graph_predict = tracer.trace(
    #     model)
    # model.mode='loss'
    # graph_loss = tracer.trace(model)
    # graph_module_loss = GraphModule(model, graph_loss, model.__class__.__name__)
    # graph_module_predict = GraphModule(model, graph_predict, model.__class__.__name__)
    # print(graph_module_loss)
    # print(graph_module_predict)

    # print(graph_module_loss.model.backbone.conv1.weight)

    # print(graph_module_loss.model.backbone.conv1.weight)
    # print(graph_module_predict.model.backbone.conv1.weight)

    # fused = fuse(graph_module_loss, True)

    # with torch.no_grad():
    #     for para in fused.parameters():
    #         para -= 1.

    # print([m for m in fused.model.backbone.conv1.modules()][1].weight)
    # print(graph_module_predict.model.backbone.conv1.weight)
    
    
    # # MMGraphModule
    # graph_module = custom_trace(model, customed_skipped_method=model.customed_skipped_method)
    # # print('predict: ', graph_module_predict, graph_module_predict(dummy_input, data_samples))
    # graph_module.to_folder('./graph_model')
    # print('loss: ', graph_module, graph_module(dummy_input, data_samples))

    # graph_module.to_mode('predict')
    # print('predict: ', graph_module, graph_module(dummy_input, data_samples))
    # print(model.data_preprocessor)
    # print('model: ', model.train_step)
    # # print(graph_module.data_preprocessor)
    # print('graph module', graph_module.train_step)




if __name__ == '__main__':
    main()
