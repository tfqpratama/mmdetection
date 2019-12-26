from .registry import DATASETS
from .xml_style_custom import XMLCustomDataset


@DATASETS.register_module
class SoybeanDataset(XMLCustomDataset):

    CLASSES = ('flower', 'seed')

    def __init__(self, **kwargs):
        super(SoybeanDataset, self).__init__(**kwargs)
