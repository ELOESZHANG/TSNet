import torch
import torch.nn as nn
import torch.utils.data as Data
import cv2
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from basicseg.test_model import Test_model
from basicseg.utils.yaml_options import parse_options, dict2str
from basicseg.utils.path_utils import *
from basicseg.data import build_dataset
from metric import PD_FA, ROCMetric, mIoU
from metric import SigmoidMetric,SamplewiseSigmoidMetric


def init_dataset(opt):
    test_opt = opt['dataset']['test']
    testset = build_dataset(test_opt)
    return testset


def init_dataloader(opt, testset):
    test_loader = Data.DataLoader(dataset=testset, batch_size=opt['exp']['bs'],
                                  sampler=None, num_workers=opt['exp'].get('nw', 8))
    return test_loader


def tensor2img(inp):
    # [b,1,h,w] -> [b,h,w]-> cpu -> numpy.array -> np.uint8
    # we don't do binarize here,
    # if you want to only contain 0 and 255, you can modify code here
    inp = torch.sigmoid(inp) * 255.
    inp = inp.squeeze(1).cpu().numpy().astype(np.uint8)
    return inp


def save_batch_img(imgs, img_names, dire):
    for i in range(len(imgs)):
        img = imgs[i]
        img_name = img_names[i]
        img_path = os.path.join(dire, img_name)
        cv2.imwrite(img_path, img)


def main():
    opt = parse_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['exp']['device'])
    # init dataset
    testset = init_dataset(opt)
    test_loader = init_dataloader(opt, testset)
    # 初始化 模型参数, 包含 网络 优化器 损失函数 学习率准则
    # initialize parameters including network, optimizer, loss function, learning rate scheduler
    model = Test_model(opt)
    save_dir = opt['exp'].get('save_dir', False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # load model params
    if opt.get('resume'):
        if opt['resume'].get('net_path'):
            model.load_network(model.net, opt['resume']['net_path'])
            print(f'load pretrained network from: {opt["resume"]["net_path"]}')
    # 一般的train(True)模式，使用Dropout和BatchNorm，而eval() Dropout和BatchNorm则不会"工作"。
    model.net.eval()
    # tqdm模块是python进度条库
    res = []
    # print(len(test_loader))
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    miou = mIoU(1)
    tmp2 = SigmoidMetric()
    tmp = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    for idx, data in enumerate(tqdm(test_loader)):
        img, label, img_name = data
        # print(label.shape)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            pred = model.test_one_iter((img, label))
            torch.cuda.synchronize()
            end = time.time()
            res.append(end - start)
        if save_dir:
            img_np = tensor2img(pred)
            label_np = label.cpu().numpy().astype(np.uint8)
            # print(img_np.shape)
            # cv2.imshow("imh",label_np[0])
            # tmp3 = img[0].numpy()
            # # print(tmp3.shape)
            tmp3 = img_np.transpose(1,2,0)
            # print(tmp3.shape)
            p = os.path.join('./jsx',img_name[0])
            # print(p)
            # save_image(tmp,os.path.join('.','jsx',img_name[0]))
            cv2.imwrite(p, tmp3)
            roc.update(pred.cpu(), label)
            pdfa.update(pred.cpu(), label)
            miou.update(pred.cpu(), label)
            tmp2.update(pred.cpu(), label)
            tmp.update(pred.cpu(), label)
            save_batch_img(img_np, img_name, save_dir)
            ture_positive_rate, false_positive_rate, recall, precision,F1_score = roc.get()
            print(img_name,"ture_positive_rate",ture_positive_rate[5],",false_positive_rate",false_positive_rate[5],",recall",recall[5],",precision",precision[5],"F1_score",F1_score)
    fa, pd = pdfa.get(len(test_loader))
    time_sum = 0
    for i in res:
        time_sum += i
    print("fa",fa[5])
    print("pd",pd[5])
    # print("miou",miou.get()[1])
    # # print(iou.get()[1])
    # print("niou",tmp.get()[1])
    print("f1-score(F1-score是F-measure在β=1时的特殊情况)",F1_score)
    # print("time_sum: %f" % time_sum)
    # print("len(res):%f" % len(res))
    # print("FPS: %f" % (1.0 / (time_sum / len(res))))
    test_mean_metric = model.get_mean_metric()
    test_norm_metric = model.get_norm_metric()
    ########## trainging done ##########
    print(f"best_mean_metric: [miou: {test_mean_metric['iou']:.4f}] [mfscore: {test_mean_metric['fscore']:.4f}]")
    print(f"best_norm_metric: [niou: {test_norm_metric['iou']:.4f}] [nfscore: {test_norm_metric['fscore']:.4f}]")


if __name__ == '__main__':
    main()
