"""电力负荷预测数据集模块。

本文件负责将原始电力数据转换为神经网络可以直接使用的监督学习样本。
主要处理流程如下：

1. 从 dealt_data 目录读取源域或目标域 CSV 文件；
2. 使用 MinMaxScaler 将各个特征缩放到 [0, 1]；
3. 使用滑动窗口构造监督学习数据；
4. 将前 n_hours 个时间点作为输入，将下一个时间点的负荷作为标签；
5. 根据 data_type 返回训练集或测试集；
6. 通过 unsqueeze 参数决定是否增加 CNN 所需的通道维。

常见张量形状：

    普通模型输入：[样本数, n_hours, n_features]
    CNN 输入：   [样本数, 1, n_hours, n_features]
    标签：       [样本数, 1]

其中 src 表示源域，tar 表示目标域。源域默认使用 90% 数据训练，
目标域使用 running_day 天训练、testing_day 天测试。
"""

import torch  # 导入 PyTorch，用于创建张量
from torch.utils.data import Dataset, DataLoader  # 导入数据集基类和批量加载器
from pathlib import Path  # 导入跨平台路径处理工具

from pandas import read_csv  # 导入 CSV 文件读取函数
from pandas import DataFrame  # 导入 DataFrame，用于构造监督学习表格
from pandas import concat  # 导入按列拼接 DataFrame 的函数

from sklearn.preprocessing import MinMaxScaler  # 导入最小最大归一化工具


class Power_ds(Dataset):
    """电力负荷预测数据集。

    该类将多变量时间序列转换成滑动窗口样本，并兼容源域和目标域
    两种数据划分方式。每次取样返回一个输入窗口和对应的负荷标签。
    """

    def __init__(self, n_hours, n_features, data_type, domain_type,
                 src_domain='panama_power', tar_domain='tetouan_region1_power', running_day=7, testing_day=30, unsqueeze=False):

        super(Power_ds, self).__init__()  # 初始化 Dataset 父类

        self.n_hours = n_hours  # 保存输入窗口包含的历史时间点数量
        self.n_features = n_features  # 保存每个时间点包含的特征数量
        self.data_type = data_type  # 保存当前数据类型：train 或 test
        self.domain_type = domain_type  # 保存当前领域类型：src 或 tar
        self.src_domain = src_domain  # 保存源域 CSV 文件名
        self.tar_domain = tar_domain  # 保存目标域 CSV 文件名
        self.running_day = running_day  # 保存目标域训练数据的天数
        self.testing_day = testing_day  # 保存目标域测试数据的天数
        self.unsqueeze = unsqueeze  # 保存是否增加 CNN 所需通道维的开关


        if (self.src_domain is not None) and (domain_type == 'src'):  # 当配置了源域且当前领域为源域时执行

            self.values, self.scaler = self.preprocessing(self.domain_type)  # 读取、归一化并构造源域监督数据

            if self.data_type == 'train':  # 当请求源域训练集时执行
                n_hours = self.n_hours  # 读取历史窗口长度
                n_features = self.n_features  # 读取每个时间点的特征数量
                n_train_samples = int(len(self.values) * 0.9)  # 使用源域前90%的样本训练
                train = self.values[:n_train_samples, :]  # 截取源域训练数据
                n_obs = n_hours * n_features  # 计算一个输入窗口包含的数值总数
                train_x, train_y = train[:, :n_obs], train[:, -n_features]  # 分离输入窗口和当前时刻负荷标签
                train_x = train_x.reshape(  # 将输入恢复为时间和特征两个维度
                    (train_x.shape[0], n_hours, n_features))  # 得到[样本数, 历史长度, 特征数]
                if self.unsqueeze:  # 如果模型需要二维卷积输入则增加通道维
                    train_x = train_x.reshape(  # 重塑为CNN需要的四维输入
                        (train_x.shape[0], 1, n_hours, n_features))  # 得到[样本数, 1, 历史长度, 特征数]
                train_y = train_y.reshape(-1, 1)  # 将标签整理为[样本数, 1]
                train_x = torch.Tensor(train_x)  # 将 NumPy 数组转换为 PyTorch 浮点张量
                train_y = torch.Tensor(train_y)  # 将标签转换为 PyTorch 浮点张量
                self.src_data_x = train_x  # 保存源域训练输入
                self.src_data_y = train_y  # 保存源域训练标签

            if self.data_type == 'test':  # 当请求源域测试集时执行
                n_hours = self.n_hours  # 读取历史窗口长度
                n_features = self.n_features  # 读取每个时间点的特征数量
                n_train_samples = int(len(self.values) * 0.9)  # 计算源域训练样本数量
                test = self.values[n_train_samples:, :]  # 截取源域后10%的测试数据
                n_obs = n_hours * n_features  # 计算一个输入窗口包含的数值总数
                test_x, test_y = test[:, :n_obs], test[:, -n_features]  # 分离测试输入和测试标签
                test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))  # 恢复为[样本数, 历史长度, 特征数]
                if self.unsqueeze:  # 如果模型需要二维卷积输入则增加通道维
                    test_x = test_x.reshape((test_x.shape[0], 1, n_hours, n_features))  # 得到[样本数, 1, 历史长度, 特征数]
                test_y = test_y.reshape(-1, 1)  # 将标签整理为[样本数, 1]
                test_x = torch.Tensor(test_x)  # 将输入转换为 PyTorch 浮点张量
                test_y = torch.Tensor(test_y)  # 将标签转换为 PyTorch 浮点张量
                self.src_data_x = test_x  # 保存源域测试输入
                self.src_data_y = test_y  # 保存源域测试标签


        if (self.tar_domain is not None) and (domain_type == 'tar'):  # 当配置了目标域且当前领域为目标域时执行

            self.values, self.scaler = self.preprocessing(self.domain_type)  # 读取、归一化并构造目标域监督数据

            if self.data_type == 'train':  # 当请求目标域训练集时执行
                n_hours = self.n_hours  # 读取历史窗口长度
                n_features = self.n_features  # 读取每个时间点的特征数量
                n_train_samples = int(self.running_day * 24)  # 将训练天数转换为训练小时数
                train = self.values[:n_train_samples, :]  # 截取目标域训练数据
                n_obs = n_hours * n_features  # 计算一个输入窗口包含的数值总数
                train_x, train_y = train[:, :n_obs], train[:, -n_features]  # 分离输入窗口和当前时刻负荷标签
                train_x = train_x.reshape(  # 将输入恢复为时间和特征两个维度
                    (train_x.shape[0], n_hours, n_features))  # 得到[样本数, 历史长度, 特征数]
                if self.unsqueeze:  # 如果模型需要二维卷积输入则增加通道维
                    train_x = train_x.reshape(  # 重塑为CNN需要的四维输入
                        (train_x.shape[0], 1, n_hours, n_features))  # 得到[样本数, 1, 历史长度, 特征数]
                train_y = train_y.reshape(-1, 1)  # 将标签整理为[样本数, 1]
                train_x = torch.Tensor(train_x)  # 将输入转换为 PyTorch 浮点张量
                train_y = torch.Tensor(train_y)  # 将标签转换为 PyTorch 浮点张量
                self.tar_data_x = train_x  # 保存目标域训练输入
                self.tar_data_y = train_y  # 保存目标域训练标签

            if self.data_type == 'test':  # 当请求目标域测试集时执行
                n_hours = self.n_hours  # 读取历史窗口长度
                n_features = self.n_features  # 读取每个时间点的特征数量
                n_train_samples = int(self.running_day * 24)  # 计算目标域训练小时数
                test = self.values[n_train_samples+1:, :]  # 从训练区间之后截取目标域测试数据
                n_obs = n_hours * n_features  # 计算一个输入窗口包含的数值总数
                test_x, test_y = test[:, :n_obs], test[:, -n_features]  # 分离测试输入和测试标签
                test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))  # 恢复为[样本数, 历史长度, 特征数]
                if self.unsqueeze:  # 如果模型需要二维卷积输入则增加通道维
                    test_x = test_x.reshape((test_x.shape[0], 1, n_hours, n_features))  # 得到[样本数, 1, 历史长度, 特征数]
                test_y = test_y.reshape(-1, 1)  # 将标签整理为[样本数, 1]
                test_x = torch.Tensor(test_x)  # 将输入转换为 PyTorch 浮点张量
                test_y = torch.Tensor(test_y)  # 将标签转换为 PyTorch 浮点张量
                self.tar_data_x = test_x  # 保存目标域测试输入
                self.tar_data_y = test_y  # 保存目标域测试标签


    def get_scaler(self):
        return self.scaler  # 返回当前数据集使用的归一化器


    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]  # 获取输入数据的特征数量

        df = DataFrame(data)  # 将数组转换为DataFrame以便进行时间偏移
        cols, names = list(), list()  # 分别保存各时间步数据和列名
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):  # 从最早的历史时间点循环到最近的历史时间点
            cols.append(df.shift(i))  # 将整张表向下移动i行，表示t-i时刻
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]  # 生成历史特征列名
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):  # 循环构造当前时刻及未来时刻的标签候选
            cols.append(df.shift(-i))  # 向上移动i行，表示t+i时刻
            if i == 0:  # 当前时刻使用t作为列名
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]  # 生成当前时刻特征列名
            else:  # 未来时刻在列名中记录偏移量
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]  # 生成未来特征列名
        # put it all together
        agg = concat(cols, axis=1)  # 将历史和未来数据按列拼接起来
        agg.columns = names  # 为拼接后的DataFrame设置描述性列名
        # drop rows with NaN values
        if dropnan:  # 如果需要删除因移动产生的缺失值
            agg.dropna(inplace=True)  # 删除包含NaN的行
        return agg  # 返回监督学习格式的DataFrame

    def preprocessing(self, domain_type):
        n_hours = self.n_hours  # 读取历史窗口长度
        n_features = self.n_features  # 读取特征数量
        running_day = self.running_day  # 读取目标域训练天数
        testing_day = self.testing_day  # 读取目标域测试天数
        data_dir = Path(__file__).resolve().parent / 'dealt_data'  # 定位项目中的数据目录

        if domain_type == 'src':  # 当处理源域数据时执行
            dataset = read_csv(  # 读取源域CSV文件
                data_dir / (self.src_domain + '.csv'),  # 拼接源域文件路径
                header=0, index_col=0)  # 第一行为表头并将第一列作为索引
            values = dataset.values  # 提取DataFrame中的数值数组
            values = values.astype('float32')  # 将数据转换为32位浮点数

            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)  # 在源域数据上拟合归一化器
            scaled = scaler.fit_transform(values)  # 将源域各特征缩放到[0,1]

        elif domain_type == 'tar':  # 当处理目标域数据时执行
            dataset = read_csv(  # 读取目标域CSV文件
                data_dir / (self.tar_domain + '.csv'),  # 拼接目标域文件路径
                header=0, index_col=0)  # 第一行为表头并将第一列作为索引
            values = dataset.values  # 提取DataFrame中的数值数组
            sample_num = running_day*24 + testing_day*24 + n_hours + 1  # 计算目标域训练、测试和窗口所需数据量
            values = values[-sample_num:, :]  # 保留末尾指定数量的数据
            values = values.astype('float32')  # 将数据转换为32位浮点数

            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)  # 在目标域选定数据上拟合归一化器
            scaled = scaler.fit_transform(values)  # 将目标域各特征缩放到[0,1]

        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, n_in=n_hours, n_out=1)  # 用历史窗口和当前时刻构造监督数据

        values = reframed.values  # 将监督DataFrame转换为NumPy数组

        return values, scaler  # 返回监督数据和归一化器

    def __len__(self):

        if self.domain_type == 'src':  # 如果当前数据集属于源域
            data = self.src_data_y  # 读取源域标签以确定样本数量

        elif self.domain_type == 'tar':  # 如果当前数据集属于目标域
            data = self.tar_data_y  # 读取目标域标签以确定样本数量

        return len(data)  # 返回数据集中的样本数量

    def __getitem__(self, item):

        if self.domain_type == 'src':  # 如果当前数据集属于源域
            data_x, data_y = self.src_data_x[item], self.src_data_y[item]  # 按索引取出源域输入和标签

        elif self.domain_type == 'tar':  # 如果当前数据集属于目标域
            data_x, data_y = self.tar_data_x[item], self.tar_data_y[item]  # 按索引取出目标域输入和标签

        return data_x, data_y  # 返回一个监督学习样本


def main():

    n_hours = 5  # 设置每个样本使用过去5个小时的数据
    running_day = 7  # 设置目标域使用7天数据进行训练
    testing_day = 30  # 设置目标域使用30天数据进行测试

    src_domain = 'tetouan_region2_power'  # 指定源域数据文件名
    tar_domain = 'tetouan_region1_power'  # 指定目标域数据文件名

    src_train_ds = Power_ds(n_hours, 4, data_type='train',  # 创建源域训练数据集
                            domain_type='src', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True)  # 增加CNN通道维

    src_test_ds = Power_ds(n_hours, 4, data_type='test',  # 创建源域测试数据集
                           domain_type='src', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True)  # 增加CNN通道维

    tar_train_ds = Power_ds(n_hours, 4, data_type='train',  # 创建目标域训练数据集
                            domain_type='tar', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True, running_day=running_day, testing_day=testing_day)  # 设置目标域训练和测试天数

    tar_test_ds = Power_ds(n_hours, 4, data_type='test',  # 创建目标域测试数据集
                           domain_type='tar', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True, running_day=running_day, testing_day=testing_day)  # 设置目标域训练和测试天数

    print("源域数据集大小：训练集 =", len(src_train_ds), "测试集 =", len(src_test_ds))  # 打印源域训练集和测试集大小

    print("目标域数据集大小：训练集 =", len(tar_train_ds), "测试集 =", len(tar_test_ds))  # 打印目标域训练集和测试集大小


    src_train_loader = DataLoader(src_train_ds, batch_size=32, shuffle=False, drop_last=False)  # 按32个样本一批加载源域训练数据
    src_test_loader = DataLoader(src_test_ds, batch_size=len(src_test_ds), shuffle=False, drop_last=False)  # 一次加载全部源域测试数据

    tar_train_loader = DataLoader(tar_train_ds, batch_size=32, shuffle=False, drop_last=False)  # 按32个样本一批加载目标域训练数据
    tar_test_loader = DataLoader(tar_test_ds, batch_size=len(tar_test_ds), shuffle=False, drop_last=False)  # 一次加载全部目标域测试数据

    #打印空行
    print()  # 输出空行以便区分不同打印结果

    x, y = next(iter(src_train_loader))  # 取出一个源域训练批次
    print("源域训练数据的形状：输入 x =", x.shape, "标签 y =", y.shape)  # 打印源域批次输入和标签形状

    x, y = next(iter(tar_train_loader))  # 取出一个目标域训练批次
    print("目标域训练数据的形状：输入 x =", x.shape, "标签 y =", y.shape)  # 打印目标域批次输入和标签形状


if __name__ == '__main__':  # 只有直接运行此文件时才执行示例主函数
    main()  # 调用示例函数检查数据集构造和形状
