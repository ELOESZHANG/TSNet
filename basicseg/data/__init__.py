import importlib
from os import path as osp
from copy import deepcopy
from basicseg.utils.path_utils import scandir
from basicseg.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicseg.data.{file_name}')
    for file_name in arch_filenames
]


def build_dataset(opt):
    # 防止变量间相互干扰
    opt = deepcopy(opt)
    # print(opt['type'])
    # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
    network_type = opt.pop('type')
    # 通过key(network_type) 去查找对应的类（在注册器中），找到对应的类，然后实例化，最终将配置描述的模型，构建出来
    net = DATASET_REGISTRY.get(network_type)(opt)
    # logger = get_root_logger()
    # logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


def main():
    opt = {'data_root': "./dataset/SIRST/trainval", 'type': 'Dataset_test', 'imgsz': 512}
    dataset = build_dataset(opt)
    img, mask = (dataset.__getitem__(0))
    print(img.shape, mask.shape)


if __name__ == '__main__':
    main()
