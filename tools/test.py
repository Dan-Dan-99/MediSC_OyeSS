# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

from extract_kp_to_txt import process_multiple_items

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument('--config2', help='second model config file path')
    parser.add_argument('--checkpoint2', help='second model checkpoint file')
    parser.add_argument('--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument('--dump', type=str, help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument('--csv-out', action='store_true', help='save predictions as CSV files')
    parser.add_argument('--csv-decimal', type=int, default=4, help='decimal places for CSV output (default: 4)')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--badcase',
        action='store_true',
        help='whether analyze badcase in test')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # -------------------- visualization --------------------
    if (args.show and not args.badcase) or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = False \
            if args.badcase else args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- badcase analyze --------------------
    if args.badcase:
        assert 'badcase' in cfg.default_hooks, \
            'BadcaseAnalyzeHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`badcase=dict(type="BadcaseAnalyzeHook")`'

        cfg.default_hooks.badcase.enable = True
        cfg.default_hooks.badcase.show = args.show
        if args.show:
            cfg.default_hooks.badcase.wait_time = args.wait_time
        cfg.default_hooks.badcase.interval = args.interval

        metric_type = cfg.default_hooks.badcase.get('metric_type', 'loss')
        if metric_type not in ['loss', 'accuracy']:
            raise ValueError('Only support badcase metric type'
                             "in ['loss', 'accuracy']")

        if metric_type == 'loss':
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.model.head.loss
        else:
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.test_evaluator

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

class MultiModelCSVHook(Hook):
    """다중 모델 CSV 저장을 위한 Hook"""
    
    def __init__(self, decimal_places=4):
        self.decimal_places = decimal_places
        self.model1_predictions = []
        self.model2_predictions = []
        self.current_model = 1
    
    def set_current_model(self, model_num):
        """현재 처리 중인 모델 번호 설정"""
        self.current_model = model_num
    
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """각 테스트 iteration 후에 예측 결과 수집"""
        if outputs is not None:
            for output in outputs:
                if hasattr(output, 'pred_instances'):
                    import torch
                    from types import SimpleNamespace
                    
                    # 안전한 numpy 변환 함수
                    def to_numpy(data):
                        if isinstance(data, torch.Tensor):
                            return data.cpu().numpy()
                        elif isinstance(data, np.ndarray):
                            return data
                        else:
                            return np.array(data)
                    
                    # 딕셔너리를 객체로 변환
                    converted_data = SimpleNamespace()
                    converted_data.img_path = getattr(output, 'img_path', '')
                    converted_data.category_id = getattr(output, 'category_id', 0)
                    
                    # pred_instances도 객체로 변환
                    pred_instances = SimpleNamespace()
                    pred_instances.keypoints = to_numpy(output.pred_instances.keypoints)
                    pred_instances.keypoint_scores = to_numpy(output.pred_instances.keypoint_scores)
                    converted_data.pred_instances = pred_instances
                    
                    if self.current_model == 1:
                        self.model1_predictions.append(converted_data)
                    else:
                        self.model2_predictions.append(converted_data)
    
    def after_test_epoch(self, runner, metrics=None):
        """테스트 완료 후 CSV 저장"""
        if self.current_model == 2 and self.model1_predictions and self.model2_predictions:
            try:
                total_preds = len(self.model1_predictions) + len(self.model2_predictions)
                print(f"Saving {total_preds} predictions from 2 models to CSV...")
                created_files = process_multiple_items(
                    [self.model1_predictions, self.model2_predictions]
                )
                print(f"CSV files created: {created_files}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
                import traceback
                traceback.print_exc()

def main():
    args = parse_args()

    # load config
    cfg1 = Config.fromfile(args.config)
    cfg1 = merge_args(cfg1, args)

    # CSV Hook 초기화 (두 모델 공용)
    csv_hook = None
    if args.csv_out:
        csv_hook = MultiModelCSVHook(decimal_places=args.csv_decimal)
        print(f"Multi-model CSV output enabled with {args.csv_decimal} decimal places")
    
    # 첫 번째 모델 실행
    print("Running inference with Model 1...")
    runner1 = Runner.from_cfg(cfg1)
    
    if csv_hook:
        csv_hook.set_current_model(1)
        runner1.register_hook(csv_hook, 'LOWEST')
    
    runner1.test()
    
    # 두 번째 모델이 있는 경우
    if args.config2 and args.checkpoint2:
        print("Running inference with Model 2...")
        cfg2 = Config.fromfile(args.config2)
        cfg2.launcher = args.launcher
        cfg2.load_from = args.checkpoint2
        cfg2.work_dir = cfg1.work_dir  # 같은 작업 디렉토리 사용
        
        if args.show_dir is not None:
            cfg2.default_hooks.visualization.enable = True
            cfg2.default_hooks.visualization.show = args.show
            cfg2.default_hooks.visualization.out_dir = args.show_dir + '_model2'  # 구분을 위해 다른 폴더명
            cfg2.default_hooks.visualization.interval = args.interval
            if args.show:
                cfg2.default_hooks.visualization.wait_time = args.wait_time

        runner2 = Runner.from_cfg(cfg2)
        
        if csv_hook:
            csv_hook.set_current_model(2)
            runner2.register_hook(csv_hook, 'LOWEST')
        
        runner2.test()
    else:
        # 단일 모델인 경우 바로 CSV 저장
        if csv_hook and csv_hook.model1_predictions:
            try:
                print(f"Saving {len(csv_hook.model1_predictions)} predictions from single model to CSV...")
                created_files = process_multiple_items(
                    csv_hook.model1_predictions,
                    #decimal_places=csv_hook.decimal_places
                )
                print(f"CSV files created: {created_files}")
            except Exception as e:
                print(f"Error saving CSV: {e}")


if __name__ == '__main__':
    main()
