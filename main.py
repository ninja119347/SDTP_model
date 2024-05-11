## The experimental results are further analyzed here.

import argparse
from exp.exp_main import Exp_Main
# import
import numpy as np
from utils.metrics import  MAPE
# parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
#
# # basic config
# parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
# parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
# parser.add_argument('--model', type=str, required=True, default='Autoformer',
#                     help='model name, options: [Autoformer, Informer, Transformer]')
#
# # data loader
# parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
# parser.add_argument('--root_path', type=str, default='./ETT-small/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
# parser.add_argument('--features', type=str, default='M',
#                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# parser.add_argument('--freq', type=str, default='h',
#                     help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#
# # forecasting task
# parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
# parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#
# # model define
# parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
# parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
#
# # optimization
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--itr', type=int, default=1, help='experiments times')
# parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
# parser.add_argument('--des', type=str, default='test', help='exp description')
# parser.add_argument('--loss', type=str, default='mse', help='loss function')
# parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#
# # GPU
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#
# # args = parser.parse_args()
# args = parser.parse_args(['--is_training', '1', '--model', 'Autoformer', '--data', 'custom',
#                           '--model_id', 'test1000', '--data_path', 'price_prediction.csv', '--enc_in', '8',
#                           '--dec_in', '8', '--c_out', '8', '--freq', 'd', '--target', 'closeIndex',
#                           '--train_epochs', '100', '--batch_size', '64', '--features', 'MS',
#                           '--seq_len', '5', '--label_len', '5', '--pred_len', '1', '--dropout', '0.01',
#                           '--learning_rate', '0.001', '--patience', '50', '--moving_avg', '5', '--lradj', 'type1',
#                           '--e_layers', '3', '--d_layers', '2', '--d_ff', '1024'])
# # setting = 'SZstock_Autoformer_custom_ftMS_sl10_ll10_pl1_dm512_nh8_el3_dl2_df1024_fc1_ebtimeF_dtTrue_test_0'
# setting = 'FCHI_Autoformer_custom_ftMS_sl5_ll5_pl1_dm512_nh8_el3_dl2_df2048_fc1_ebtimeF_dtTrue_test_1'
# Exp = Exp_Main
# exp = Exp(args)
# exp.predict(setting, True)
path1 = './figure_data/'
path2 ='./results/'
setting ="test_Autoformer_custom_ftMS_sl5_ll5_pl1_dm512_nh8_el2_dl2_df1024_fc1_ebtimeF_dtTrue_test_1"
preds = np.load(path2+setting+'/pred.npy')
trues = np.load(path2+setting+'/true.npy')
metrics = np.load(path2+setting+'/metrics.npy')
# setting1 ="SZ"
# preds1 = np.load(path+setting1+'/pred.npy')
# trues1 = np.load(path+setting1+'/true.npy')
# metrics1 = np.load(path+setting1+'/metrics.npy')
# setting2 ="HSI"
# preds2 = np.load(path+setting2+'/pred.npy')
# trues2 = np.load(path+setting2+'/true.npy')
# metrics2 = np.load(path+setting2+'/metrics.npy')
print(MAPE(preds,trues))
print(preds.shape, trues.shape, metrics.shape)
print(metrics)
import numpy as np
# trues = trues.reshape(1000, 1)
# prediction = np.load('./results/'+setting+'/real_prediction.npy')
# import matplotlib.pyplot as plt
# import matplotlib
# # print(prediction.shape)
# # print(prediction)
# # import seaborn as sns
# fig =plt.figure(figsize=(25, 15))
# ax1 = fig.add_subplot(3, 1, 1)  # 主图
# ax2 = fig.add_subplot(3, 1, 2)  # 子图
# ax3 = fig.add_subplot(3, 1, 3)  # 子图
# # ax1 = plt.gca()
#
# # 设置坐标轴的字体大小
# ax1.tick_params(axis='both', which='both', labelsize=20)
# ax1.set_ylabel('SH',fontsize=25)
#
# # matplotlib.use('TKAgg')
# # plt.plot(prediction[:,:,-1],label='prediction')
# # ylabel('Displacement (m)')
# # xlabel('Times(s)','fontsize',12,'fontname','Times','FontWeight','bold')
# ax1.plot(trues[:, -1], label='GroundTruth',linewidth=2.5)
# ax1.plot(preds[:, -1], label='Prediction',linewidth=2.5)
# xticks=[0 ,500 ,1000]
# xticklabels={'07/25/2016','08/08/2018','06/30/2020'}
# ax1.set_xticks(xticks, xticklabels)
# ax1.legend(fontsize=20)
#
# ax2.tick_params(axis='both', which='both', labelsize=20)
# ax2.set_ylabel('SZ',fontsize=25)
#
# # matplotlib.use('TKAgg')
# # plt.plot(prediction[:,:,-1],label='prediction')
# # ylabel('Displacement (m)')
# # xlabel('Times(s)','fontsize',12,'fontname','Times','FontWeight','bold')
# ax2.plot(trues1[:, -1], label='GroundTruth',linewidth=2.5)
# ax2.plot(preds1[:, -1], label='Prediction',linewidth=2.5)
# xticks=[0 ,500 ,1000]
# xticklabels1={'/04/18/2018','09/30/2020','06/03/2020'}
# ax2.set_xticks(xticks, xticklabels1)
# ax2.legend(fontsize=20)
#
# ax3.tick_params(axis='both', which='both', labelsize=20)
# ax3.set_ylabel('HSI',fontsize=25)
#
# # matplotlib.use('TKAgg')
# # plt.plot(prediction[:,:,-1],label='prediction')
# # ylabel('Displacement (m)')
# # xlabel('Times(s)','fontsize',12,'fontname','Times','FontWeight','bold')
# ax3.plot(trues2[:, -1], label='GroundTruth',linewidth=2.5)
# ax3.plot(preds2[:, -1], label='Prediction',linewidth=2.5)
# xticks=[0 ,500 ,1000]
# xticklabels={'05/10/2018','05/21/2020','05/31/2022'}
# ax3.set_xticks(xticks, xticklabels)
# ax3.legend(fontsize=20)
# # 设置 x 轴刻度位置和标签
# # plt.xticks(years)
# # plt.savefig(path+'/HSI'+'/result.png')
# fig.subplots_adjust(hspace=0.3)
# plt.savefig('./figure_data/img.eps', dpi=300)  # eps文件，用于LaTeX
# plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
