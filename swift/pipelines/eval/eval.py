# Copyright (c) ModelScope Contributors. All rights reserved.
import csv
import json
import os
import re
from contextlib import nullcontext
from evalscope.constants import EvalBackend, EvalType
from evalscope.run import TaskConfig, run_task
from evalscope.summarizer import Summarizer
from typing import List, Optional, Union

from swift.arguments import EvalArguments
from swift.dataset import MediaResource
from swift.utils import append_to_jsonl, get_logger
from ..base import SwiftPipeline
from ..infer import run_deploy

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        eval_report = {}
        deploy_context = nullcontext() if args.eval_url else run_deploy(args, return_url=True)
        with deploy_context as base_url:
            base_url = args.eval_url or base_url

            task_cfg = self.get_task_cfg(args.eval_dataset, args.eval_backend, base_url)
            result = self.get_task_result(task_cfg)
            eval_report[args.eval_backend] = result

            lcb_sample_csv = self._export_live_code_bench_sample_csv(task_cfg)
            if lcb_sample_csv is not None:
                eval_report['live_code_bench_sample_csv'] = lcb_sample_csv

        eval_report.update({
            'time': args.time,
            'model': args.model,
            'adapters': args.adapters,
            'result_path': args.result_path,
            'eval_output_dir': args.eval_output_dir,
            'eval_limit': args.eval_limit
        })

        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'The eval result have been saved to result_jsonl: `{args.result_jsonl}`.')
        return eval_report

    @staticmethod
    def _safe_json_loads(content):
        if not isinstance(content, str):
            return None
        try:
            return json.loads(content)
        except Exception:
            return None

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        content = content.strip()
        if content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
            if '\n' in content:
                first_line, rest = content.split('\n', 1)
                if re.match(r'^[A-Za-z0-9_+-]+$', first_line.strip()):
                    return rest.strip()
            return content
        return content

    @classmethod
    def _extract_code_block(cls, sample_score: dict) -> str:
        score = sample_score.get('score') or {}
        extracted_prediction = score.get('extracted_prediction')
        if isinstance(extracted_prediction, str) and extracted_prediction.strip():
            return cls._strip_code_fence(extracted_prediction)

        prediction = score.get('prediction')
        if not isinstance(prediction, str) or not prediction.strip():
            return ''

        blocks = re.findall(r'```(?:[A-Za-z0-9_+-]+)?\n(.*?)```', prediction, flags=re.DOTALL)
        if blocks:
            return '\n\n'.join(block.strip() for block in blocks if block.strip())

        return ''

    @staticmethod
    def _extract_sample_index(sample: dict, fallback_index: int) -> int:
        index = sample.get('index')
        if index is None:
            sample_score = sample.get('sample_score') or {}
            index = sample_score.get('sample_id')
        if isinstance(index, (int, float)):
            return int(index)
        if isinstance(index, str):
            index = index.strip()
            if index.isdigit() or (index.startswith('-') and index[1:].isdigit()):
                return int(index)
        return fallback_index

    def _extract_error_reason(self, sample_score: dict) -> str:
        score = sample_score.get('score') or {}
        metadata = score.get('metadata') or sample_score.get('metadata') or {}
        final_metadata = metadata.get('final_metadata')
        if isinstance(final_metadata, list):
            for item in final_metadata:
                if not isinstance(item, (list, tuple)):
                    item = [item]
                for error_meta in item:
                    parsed_meta = self._safe_json_loads(error_meta)
                    if isinstance(parsed_meta, dict) and parsed_meta.get('error_message'):
                        return str(parsed_meta['error_message'])
                    if isinstance(error_meta, str) and error_meta:
                        return error_meta

        explanation = score.get('explanation') or sample_score.get('explanation')
        if isinstance(explanation, str) and explanation:
            return explanation

        return 'Unknown error'

    @staticmethod
    def _is_correct(sample_score: dict) -> bool:
        score = sample_score.get('score') or {}
        value = score.get('value') or {}
        acc = value.get('acc')
        if isinstance(acc, bool):
            return acc
        if isinstance(acc, (int, float)):
            return acc >= 1.0

        metadata = score.get('metadata') or sample_score.get('metadata') or {}
        pass_rate = metadata.get('pass_rate')
        if isinstance(pass_rate, (int, float)):
            return pass_rate > 0

        return False

    def _find_latest_native_eval_dir(self) -> Optional[str]:
        native_root = os.path.join(self.args.eval_output_dir, 'native')
        if not os.path.isdir(native_root):
            return None

        candidates = []
        for dirname in os.listdir(native_root):
            path = os.path.join(native_root, dirname)
            if not os.path.isdir(path):
                continue
            review_dir = os.path.join(path, 'reviews', self.args.model_suffix)
            if os.path.isdir(review_dir):
                candidates.append(path)
        if not candidates:
            return None

        return max(candidates, key=os.path.getmtime)

    def _export_live_code_bench_sample_csv(self, task_cfg: TaskConfig) -> Optional[str]:
        if task_cfg.eval_backend != EvalBackend.NATIVE:
            return None
        if not any(dataset.lower().startswith('live_code_bench') for dataset in self.args.eval_dataset):
            return None

        run_dir = self._find_latest_native_eval_dir()
        if run_dir is None:
            logger.warning('Cannot find native eval output directory for LiveCodeBench sample CSV export.')
            return None

        review_dir = os.path.join(run_dir, 'reviews', self.args.model_suffix)
        if not os.path.isdir(review_dir):
            logger.warning(f'Review directory not found: {review_dir}')
            return None

        review_files = [
            os.path.join(review_dir, filename)
            for filename in sorted(os.listdir(review_dir))
            if filename.startswith('live_code_bench') and filename.endswith('.jsonl')
        ]
        if not review_files:
            logger.warning(f'No LiveCodeBench review file found under: {review_dir}')
            return None

        rows = []
        for review_file in review_files:
            with open(review_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        rows.append({
                            'index': line_no,
                            'correct': '0',
                            'error_reason': 'Malformed review JSON line',
                            'code_block': ''
                        })
                        continue

                    sample_score = sample.get('sample_score') or {}
                    is_correct = self._is_correct(sample_score)
                    sample_index = self._extract_sample_index(sample, fallback_index=line_no)
                    rows.append({
                        'index': sample_index,
                        'correct': '1' if is_correct else '0',
                        'error_reason': '' if is_correct else self._extract_error_reason(sample_score),
                        'code_block': self._extract_code_block(sample_score),
                    })

        if not rows:
            logger.warning('No rows found when exporting LiveCodeBench sample CSV.')
            return None

        model_name = self.args.model_suffix.replace('/', '_')
        csv_path = os.path.join(run_dir, f'{model_name}_live_code_bench_sample_results.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'correct', 'error_reason', 'code_block'])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f'LiveCodeBench per-sample CSV saved to: {csv_path}')
        return csv_path

    def get_task_result(self, task_cfg: TaskConfig):
        run_task(task_cfg=task_cfg)
        reports = Summarizer.get_report_from_cfg(task_cfg=task_cfg)
        result = {}
        if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
            for report in reports:
                if report[self.args.model_suffix] != '-':
                    result[report['dataset']] = {report['metric']: report[self.args.model_suffix]}
        elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
            for report in reports:
                splited_key = next(iter(report)).rsplit('_', 2)
                if len(splited_key) == 3:
                    _, dataset, metric = splited_key
                else:
                    dataset, metric = '-', '-'
                result[dataset] = {metric: list(report.values())[0]}
        else:
            result = reports
        return result

    def get_task_cfg(self, dataset: List[str], eval_backend: str, url: str):
        assert eval_backend in {EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT}
        if eval_backend == EvalBackend.OPEN_COMPASS:
            if self.args.local_dataset:
                if os.path.exists('data'):
                    if not os.path.exists(os.path.join('data', 'CMB')):
                        raise RuntimeError('Opencompass need a `data` folder in your work dir('
                                           'which will be created automatically by swift eval), '
                                           'but a local path named `data` already exists, '
                                           'please consider moving the dir to another location.')
                else:
                    local_dir = MediaResource.download(
                        'https://modelscope.cn/datasets/'
                        'opencompass/OpenCompassDataComplete/'
                        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
                    os.symlink(os.path.join(local_dir, 'data'), 'data')

            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        elif eval_backend == EvalBackend.VLM_EVAL_KIT:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        else:
            task_cfg = self.get_native_task_cfg(dataset, url)
        return task_cfg

    def get_native_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'native')
        return TaskConfig(
            model=args.model_suffix,
            eval_type=EvalType.SERVICE,
            api_url=url,
            api_key=args.api_key or 'EMPTY',
            datasets=dataset,
            work_dir=work_dir,
            limit=args.eval_limit,
            eval_batch_size=args.eval_num_proc,
            dataset_args=args.eval_dataset_args,
            generation_config=args.eval_generation_config,
            **args.extra_eval_args)

    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'opencompass')
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets':
                dataset,
                'batch_size':
                args.eval_num_proc,
                'work_dir':
                work_dir,
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key or 'EMPTY',
                    'is_chat': args.use_chat_template
                }],
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'vlmeval')
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data':
                dataset,
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key or 'EMPTY',
                    **args.eval_generation_config
                }],
                'nproc':
                args.eval_num_proc,
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)


def eval_main(args: Optional[Union[List[str], EvalArguments]] = None):
    return SwiftEval(args).main()
