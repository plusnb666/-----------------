import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler


class Power_ds(Dataset):
    def __init__(self, n_hours, n_features, data_type, domain_type,
                 src_domain='panama_power', tar_domain='tetouan_region1_power', running_day=7, testing_day=30, unsqueeze=False):

        super(Power_ds, self).__init__()

        self.n_hours = n_hours
        self.n_features = n_features
        self.data_type = data_type
        self.domain_type = domain_type
        self.src_domain = src_domain
        self.tar_domain = tar_domain
        self.running_day = running_day
        self.testing_day = testing_day
        self.unsqueeze = unsqueeze


        if (self.src_domain is not None) and (domain_type == 'src'):

            self.values, self.scaler = self.preprocessing(self.domain_type)

            if self.data_type == 'train':
                n_hours = self.n_hours
                n_features = self.n_features
                n_train_samples = int(len(self.values) * 0.9)
                train = self.values[:n_train_samples, :]
                n_obs = n_hours * n_features
                train_x, train_y = train[:, :n_obs], train[:, -n_features]
                train_x = train_x.reshape(
                    (train_x.shape[0], n_hours, n_features))
                if self.unsqueeze:
                    train_x = train_x.reshape(
                        (train_x.shape[0], 1, n_hours, n_features))  # [batch_size, 1, n_hours, n_features]
                train_y = train_y.reshape(-1, 1)
                train_x = torch.Tensor(train_x)
                train_y = torch.Tensor(train_y)
                self.src_data_x = train_x
                self.src_data_y = train_y

            if self.data_type == 'test':
                n_hours = self.n_hours
                n_features = self.n_features
                n_train_samples = int(len(self.values) * 0.9)
                test = self.values[n_train_samples:, :]
                n_obs = n_hours * n_features
                test_x, test_y = test[:, :n_obs], test[:, -n_features]
                test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))
                if self.unsqueeze:
                    test_x = test_x.reshape((test_x.shape[0], 1, n_hours, n_features))
                test_y = test_y.reshape(-1, 1)
                test_x = torch.Tensor(test_x)
                test_y = torch.Tensor(test_y)
                self.src_data_x = test_x
                self.src_data_y = test_y


        if (self.tar_domain is not None) and (domain_type == 'tar'):

            self.values, self.scaler = self.preprocessing(self.domain_type)

            if self.data_type == 'train':
                n_hours = self.n_hours
                n_features = self.n_features
                n_train_samples = int(self.running_day * 24)
                train = self.values[:n_train_samples, :]
                n_obs = n_hours * n_features
                train_x, train_y = train[:, :n_obs], train[:, -n_features]
                train_x = train_x.reshape(
                    (train_x.shape[0], n_hours, n_features))
                if self.unsqueeze:
                    train_x = train_x.reshape(
                        (train_x.shape[0], 1, n_hours, n_features))  # [batch_size, 1, n_hours, n_features]
                train_y = train_y.reshape(-1, 1)
                train_x = torch.Tensor(train_x)
                train_y = torch.Tensor(train_y)
                self.tar_data_x = train_x
                self.tar_data_y = train_y

            if self.data_type == 'test':
                n_hours = self.n_hours
                n_features = self.n_features
                n_train_samples = int(self.running_day * 24)
                test = self.values[n_train_samples+1:, :]
                n_obs = n_hours * n_features
                test_x, test_y = test[:, :n_obs], test[:, -n_features]
                test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))
                if self.unsqueeze:
                    test_x = test_x.reshape((test_x.shape[0], 1, n_hours, n_features))
                test_y = test_y.reshape(-1, 1)
                test_x = torch.Tensor(test_x)
                test_y = torch.Tensor(test_y)
                self.tar_data_x = test_x
                self.tar_data_y = test_y


    def get_scaler(self):
        return self.scaler


    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]

        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def preprocessing(self, domain_type):
        n_hours = self.n_hours
        n_features = self.n_features
        running_day = self.running_day
        testing_day = self.testing_day
        data_dir = Path(__file__).resolve().parent / 'dealt_data'

        if domain_type == 'src':
            dataset = read_csv(
                data_dir / (self.src_domain + '.csv'),
                header=0, index_col=0)
            values = dataset.values
            values = values.astype('float32')

            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
            scaled = scaler.fit_transform(values)

        elif domain_type == 'tar':
            dataset = read_csv(
                data_dir / (self.tar_domain + '.csv'),
                header=0, index_col=0)
            values = dataset.values
            sample_num = running_day*24 + testing_day*24 + n_hours + 1 #下X天的运行数据作为测试集
            values = values[-sample_num:, :]
            values = values.astype('float32')

            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
            scaled = scaler.fit_transform(values)

        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, n_in=n_hours, n_out=1)

        values = reframed.values

        return values, scaler

    def __len__(self):

        if self.domain_type == 'src':
            data = self.src_data_y

        elif self.domain_type == 'tar':
            data = self.tar_data_y

        return len(data)

    def __getitem__(self, item):

        if self.domain_type == 'src':
            data_x, data_y = self.src_data_x[item], self.src_data_y[item]

        elif self.domain_type == 'tar':
            data_x, data_y = self.tar_data_x[item], self.tar_data_y[item]

        return data_x, data_y


def main():

    n_hours = 5
    running_day = 7
    testing_day = 30

    src_domain = 'tetouan_region2_power'
    tar_domain = 'tetouan_region1_power'

    src_train_ds = Power_ds(n_hours, 4, data_type='train',
                            domain_type='src', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True)

    src_test_ds = Power_ds(n_hours, 4, data_type='test',
                            domain_type='src', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True)

    tar_train_ds = Power_ds(n_hours, 4, data_type='train',
                            domain_type='tar', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True, running_day=running_day, testing_day=testing_day)

    tar_test_ds = Power_ds(n_hours, 4, data_type='test',
                            domain_type='tar', src_domain=src_domain, tar_domain=tar_domain, unsqueeze=True, running_day=running_day, testing_day=testing_day)

    print("源域数据集大小：训练集 =", len(src_train_ds), "测试集 =", len(src_test_ds))

    print("目标域数据集大小：训练集 =", len(tar_train_ds), "测试集 =", len(tar_test_ds))


    src_train_loader = DataLoader(src_train_ds, batch_size=32, shuffle=False, drop_last=False)
    src_test_loader = DataLoader(src_test_ds, batch_size=len(src_test_ds), shuffle=False, drop_last=False)

    tar_train_loader = DataLoader(tar_train_ds, batch_size=32, shuffle=False, drop_last=False)
    tar_test_loader = DataLoader(tar_test_ds, batch_size=len(tar_test_ds), shuffle=False, drop_last=False)

    #打印空行
    print()

    x, y = next(iter(src_train_loader))
    print("源域训练数据的形状：输入 x =", x.shape, "标签 y =", y.shape)

    x, y = next(iter(tar_train_loader))
    print("目标域训练数据的形状：输入 x =", x.shape, "标签 y =", y.shape)


if __name__ == '__main__':
    main()
