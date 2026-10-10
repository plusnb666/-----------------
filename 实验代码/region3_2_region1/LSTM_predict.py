import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from 实验代码.typical_models import LSTM_model
import time
from 实验代码.power_dataset import Power_ds


# calculate MAPE
def mean_absolute_percentage_error(real, predict):
    res = 0
    count = 0
    for i in range(len(real)):
        if real[i] != 0:
            res += abs((predict[i] - real[i]) / real[i])
            count += 1
    if count == 0:
        return 0  # Avoid division by zero
    return float(res / count)

def plot_predict(pre, test_labels):
    plt.plot(test_labels, label='real')
    plt.plot(pre, label='predict')
    plt.title('real-predict')
    plt.xlabel('hour')
    plt.ylabel('power')
    plt.legend()  # 显示图例，与label搭配使用
    plt.show()


task = 'region3_2_tetouan_region1'

src_domain1 = 'tetouan_region3_power'
tar_domain = 'tetouan_region1_power'

n_hours = 5
n_features = 4

src_batchsize = 32
tar_batchsize = 16
src_epochs = 100
tar_epochs = 100
lr = 1e-3

running_day = 7
testing_day = 30


mape_results = []
rmse_results = []
R2_results = []
mae_results = []
runtime_results = []

max_r2 = 0
best_predict = 0

n_runs = 20


device = torch.device('cuda')

src1_train_ds = Power_ds(n_hours, n_features, data_type='train',
                        domain_type='src', src_domain=src_domain1, unsqueeze=False)


tar_train_ds = Power_ds(n_hours, n_features, data_type='train',
                        domain_type='tar', tar_domain=tar_domain, unsqueeze=False, running_day=running_day, testing_day=testing_day)

tar_test_ds = Power_ds(n_hours, n_features, data_type='test',
                       domain_type='tar', tar_domain=tar_domain, unsqueeze=False, running_day=running_day, testing_day=testing_day)

scaler = tar_test_ds.get_scaler()


src1_train_loader = DataLoader(src1_train_ds, batch_size=src_batchsize, shuffle=True)

tar_train_loader = DataLoader(tar_train_ds, batch_size=tar_batchsize, shuffle=True)
tar_test_loader = DataLoader(tar_test_ds, batch_size=len(tar_test_ds), shuffle=False)

for run in range(n_runs):
    start_time = time.time()

    model1 = LSTM_model(feature_num=n_features, layer_num=1, nodes_num=128, dropout=0.3).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)

    criteon = nn.L1Loss()


    for epoch in range(src_epochs):
        total_loss1 = []

        for step, (x, y) in enumerate(src1_train_loader):
            x, y = x.to(device), y.to(device)
            logits = model1(x)
            loss1 = criteon(logits, y)
            total_loss1.append(loss1.item())

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

        print('MAE source1 training loss of run{} ,epoch{}:{:.4}'.format(run+1, epoch+1, np.mean(total_loss1)))


    # #将源域训练好的模型参数保存在本地
    # torch.save(model.state_dict(), 'D:\\小论文\\小论文2\\实验代码\\panama2tetouan_region1\\pre_trained_model\\src_model.pth')
    #
    # #加载源域训练好的模型参数
    # model.load_state_dict(torch.load('D:\\小论文\\小论文2\\实验代码\\panama2tetouan_region1\\pre_trained_model\\src_model.pth'))

    #微调
    for epoch in range(tar_epochs):
        total_loss1 = []

        for step, (x, y) in enumerate(tar_train_loader):
            x, y = x.to(device), y.to(device)

            logits1 = model1(x)
            loss1 = criteon(logits1, y)
            total_loss1.append(loss1.item())
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()


        # if (epoch+1)%5 == 0:
        print('MAE target training loss of run{},epoch{}:{:.4} of src1'.format(run+1, epoch+1, np.mean(total_loss1)))



    for test_X, test_y in tar_test_loader:
        test_X, test_y = test_X.to(device), test_y.to(device)

        with torch.no_grad():

            yhat = model1(test_X)

        yhat = yhat.cpu().detach().numpy()
        if yhat.shape[1] != 1:
            raise ValueError("Prediction result shape is incorrect")

        # print(yhat.shape)
        test_X = test_X.cpu().detach().numpy()
        test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, -n_features+1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.cpu().detach().numpy()
        test_y = test_y.reshape(len(test_y), 1)
        inv_y = concatenate((test_y, test_X[:, -n_features+1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        rmse_results.append(rmse)

        # calculate MAPE
        mape = mean_absolute_percentage_error(inv_y, inv_yhat)
        print('Test mape: %.3f' % mape)
        mape_results.append(mape)

        # calculate R^2
        R2 = r2_score(inv_y, inv_yhat)
        print(f"R-squared (R^2): {R2:.3f}")
        R2_results.append(R2)

        # calculate MAE
        mae = mean_absolute_error(inv_y, inv_yhat)
        print('Test MAE: %.3f' % mae)
        mae_results.append(mae)

        # calculate TIME
        end_time = time.time()
        runtime = end_time - start_time
        runtime_results.append(runtime)

        print(
            f"Run {run + 1}: Runtime = {runtime:.3f} seconds, MAE = {mae:.3f}, MAPE = {mape:.3f}, RMSE = {rmse:.3f}, R2 = {R2:.3f}")

        if  R2 > max_r2:
            max_r2 = R2
            best_predict = inv_yhat


# 计算n次运行的均值
mean_runtime = np.mean(runtime_results)
mean_mae = np.mean(mae_results)
mean_mape = np.mean(mape_results)
mean_rmse = np.mean(rmse_results)
mean_R2 = np.mean(R2_results)

#打印所有的实验结果列表
print("MAE results:", mae_results)
print("MAPE results:", mape_results)
print("RMSE results:", rmse_results)
print("R2 results:", R2_results)

# 打印平均值
print('Task: {}: running days = {}, testing days = {}, n_hours = {}'.format(task, running_day, testing_day, n_hours))
print(f"Average Runtime over {n_runs} runs: {mean_runtime:.3f} seconds")
print(f"Average MAE over {n_runs} runs: {mean_mae:.3f}")
print(f"Average MAPE over {n_runs} runs: {mean_mape:.3f}")
print(f"Average RMSE over {n_runs} runs: {mean_rmse:.3f}")
print(f"Average R2 over {n_runs} runs: {mean_R2:.3f}")

# plot_predict(best_predict, inv_y)
