import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from feature_visualization import visualize_features


"""MNIST 到 MNIST-M 的领域对抗训练入口。

训练时源域使用真实数字标签，目标域只参与领域判别；梯度反转层让共享
特征同时具有类别判别性和领域不变性。该实验是本项目中基于特征的迁移
学习补充案例，可作为将 DANN 扩展到负荷预测的参考。
"""


def main():
    """加载源域/目标域数据，训练 DANN 并评估目标域准确率。"""

    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('dataset', source_dataset_name)
    target_image_root = os.path.join('dataset', target_dataset_name)
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 128
    image_size = 28
    n_epoch = 50

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # 加载源域 MNIST 和目标域 MNIST-M，并为两个域设置各自归一化方式。

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_source = datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    # 共享 CNN 特征提取器 + 类别分类器 + 领域分类器。

    my_net = CNNModel()

    # 两个域的分类损失与源域标签损失共同反向传播。

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # alpha 从接近 0 逐步增加，使领域对抗约束平滑加入训练过程。
    best_accu_t = 0.0

    s_feature_list = []
    t_feature_list = []

    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # alpha = 0

            # 源域：同时计算数字类别损失和“源域”领域损失。
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()


            class_output, domain_output, s_feature_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # 目标域：没有类别标签，只计算“目标域”领域损失。
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output, t_feature_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            if epoch == n_epoch-1:
                s_feature_list.append(s_feature_output.data.cpu().numpy())
                t_feature_list.append(t_feature_output.data.cpu().numpy())

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                     err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

        print('\n')
        accu_s = test(source_dataset_name)
        print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
        accu_t = test(target_dataset_name)
        print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')

    s_feature_list = np.concatenate(s_feature_list, axis=0)
    t_feature_list = np.concatenate(t_feature_list, axis=0)

    # 为降低 t-SNE 可视化开销，源域和目标域特征每隔 100 个样本抽取一个。
    s_feature_list = s_feature_list[::100]
    t_feature_list = t_feature_list[::100]

    visualize_features(s_feature_list, t_feature_list)



if __name__ == '__main__':
    main()
