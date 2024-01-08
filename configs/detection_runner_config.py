import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

opt = parser.parse_args()