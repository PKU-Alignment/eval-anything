import argparse

from eval_anything.pipeline.base_task import BaseTask
from eval_anything.utils.utils import parse_unknown_args


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--eval_info', type=str, required=True, default='evaluate.yaml', help="YAML filename for evaluation configurations in eval-anything/configs/.")
    parser.add_argument(
        '--eval_info',
        type=str,
        required=False,
        default='evaluate.yaml',
        help='YAML filename for evaluation configurations in eval-anything/configs/.',
    )
    known_args, unparsed_args = parser.parse_known_args()
    unparsed_args = parse_unknown_args(unparsed_args)
    task = BaseTask(known_args.eval_info, **unparsed_args)
    task.iterate_run()


if __name__ == '__main__':
    main()
