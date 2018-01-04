from .base_options import BaseOptions

class BaseAttributeOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)

        # basic options
        self.parser.add_argument('--n_attr', type = int, default = 500, help = 'number of attribute entries')

        # data files
        # refer to "scripts/preproc_inshop.py" for more information
        self.parser.add_argument('--fn_sample', type = str, default = 'Label/samples_attr.json', help = 'path of sample index file')
        self.parser.add_argument('--fn_label', type = str, default = 'Label/attribute_label_top500.json', help = 'path of attribute label file')
        self.parser.add_argument('--fn_entry', type = str, default = 'Label/attribute_entry_top500.json', help = 'path of attribute entry file')
        self.parser.add_argument('--fn_split', type = str, default = 'Label/split_attr.json', help = 'path of split file')


class TrainAttributeOptions(BaseAttributeOptions):

    def initialize(self):

        BaseAttributeOptions.initialize(self)

        # train

        # set train
        self.is_train = True

class TestAttributeOptions(BaseAttributeOptions):

    def initialize(self):

        BaseAttributeOptions.initialize(self)

        # test

        # set test
        self.is_train = False

