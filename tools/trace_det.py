# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
from mmdet.models.task_modules import AnchorGenerator
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmengine.config import Config, DictAction

from mmrazor.models.utils import (CustomTracer, FXModelWrapper,
                                  UntracedMethodRegistry,
                                  register_skipped_method)
from mmrazor.registry import MODELS
from mmrazor.utils import register_all_modules


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


class ArangeForFx(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._is_leaf_module = True

    def forward(self, x):
        return torch.arange(x)


class TestIf(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self._is_leaf_module = True

    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.test_if = TestIf()

    def forward(self, x):
        return self.test_if(x)


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

    # model = Net()
    # # tracer = CustomedTracer(customed_leaf_module=(ArangeForFx,))

    # tracer = CustomTracer()
    # # from torch.fx import Tracer
    # # tracer = Tracer
    # graph = tracer.trace(model)
    # name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    # # traced = GraphModule(tracer.root, graph, name)
    # print(graph)

    # test openmmlab
    # build model
    # register_skipped_method()
    from brevitas.fx.brevitas_tracer import ValueTracer
    _model = MODELS.build(cfg.model.model)
    model = FXModelWrapper(_model, mode='predict',
                        #    customed_skipped_method='mmdet.models.task_modules.prior_generators.AnchorGenerator.grid_priors',)
                           customed_skipped_method=['mmdet.models.AnchorHead.loss_by_feat', 'mmdet.models.AnchorHead.predict_by_feat',
                                                    'mmdet.models.SingleStageDetector.add_pred_to_datasample'],)
    tracer = CustomTracer()
    # tracer = ValueTracer()
    # model.ori_forward = model.forward
    # model.forward = model.extract_feat
    # model.forward = model.loss
    dummy_input = torch.rand(1, 3, 128, 128)
    packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
    data = model.model.data_preprocessor(packed_inputs, False)
    traced = tracer.trace(model, concrete_args={**data})
    print(traced)


if __name__ == '__main__':
    main()
