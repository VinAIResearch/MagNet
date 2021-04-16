import argparse


class BaseOptions:
    """Base argument parser"""

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Dataset config
        parser.add_argument("--dataset", required=True, type=str, help="dataset name: cityscapes, deepglobe")
        parser.add_argument("--root", type=str, default="", help="path to images for training and testing")
        parser.add_argument("--datalist", type=str, default="", help="path to .txt containing image and label path")
        parser.add_argument(
            "--scales", required=True, type=str, help="scales: w1-h1,w2-h2,... , e.g. 512-512,1024-1024,2048-2048"
        )
        parser.add_argument(
            "--crop_size", required=True, metavar="N", nargs="+", type=int, help="crop size, e.g. 256 128"
        )
        parser.add_argument(
            "--input_size", required=True, metavar="N", nargs="+", type=int, help="input size, e.g. 256 128"
        )
        parser.add_argument("--num_workers", default=1, type=int, help="number of workers for dataloader")

        # Model config
        parser.add_argument(
            "--model", required=True, type=str, help="model name. One of: fpn, psp, hrnet18+ocr, hrnet48+ocr"
        )
        parser.add_argument("--num_classes", required=True, type=int, help="number of classes")
        parser.add_argument("--pretrained", required=True, type=str, help="pretrained weight")
        parser.add_argument(
            "--pretrained_refinement",
            nargs="+",
            default=[""],
            type=str,
            help="pretrained weight (s) refinement module",
        )
        self.parser = parser

    def parse(self):
        """Parse arguments

        Returns:
            namespace: arguments after parsing
        """
        args = self.parser.parse_args()

        # Parse scales
        args.scales = [tuple(int(x) for x in s.split("-")) for s in args.scales.split(",")]

        # Convert types of crop_size and input_size
        args.crop_size = tuple(args.crop_size)
        args.input_size = tuple(args.input_size)

        # Check the number of pretrained_refinement
        if len(args.pretrained_refinement) == 1:
            args.pretrained_refinement = args.pretrained_refinement[0]

        return args
