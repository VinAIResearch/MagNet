from .base import BaseOptions

class TestOptions(BaseOptions):

    def __init__(self):
        super().__init__()
        parser = self.parser

        self.parser = parser
    
    def parse(self):
        args = super().parse()
        args.phase = "test"
        return args