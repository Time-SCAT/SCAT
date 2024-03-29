import argparse
import os
import torch
from cluster.wind_station import WindArgs
from cluster.station import Args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_new_windformer import EXP_new_windformer
import random
import numpy as np
import pandas

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, new_energy, short_term_forecast, imputation, classification, anomaly_detection, new_energy]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='STID',
                    help='model name, options: [Autoformer, Transformer, DISformer, PatchTST, DLinear, ETSformer, FEDformer, Informer, Pyraformer, Reformer, LightTS, TimesNet]')

# MICN, 有问题


# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='C_VALUE', help='taerget feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=16, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=15, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=15, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model, 512')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn, 32')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8
                    , help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# cluster
parser.add_argument('--station', type=str, default='J00018', help='station name , options: [J00018, datang]')
parser.add_argument('--nwp', type=bool, default=True, help='enable using nwp weather feature')
parser.add_argument('--real', type=bool, default=False, help='enable using real weather feature')
parser.add_argument('--picture', type=bool, default=False, help='enable print the picture of cluster result')
parser.add_argument('--find_key', type=bool, default=False, help='enable Recalculate gamma and n')
parser.add_argument('--cluster_enc_in', type=int, default=5, help='cluster encoder input size')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
elif args.task_name == 'new_energy':
    Exp = EXP_new_windformer
else:
    Exp = Exp_Long_Term_Forecast

mses = []
maes = []

if args.task_name == 'new_energy' and args.model == 'Windformer':
    print('>>>>>>>start cluster training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if args.data in['ETTm1', 'ETTm2', 'ETTm3', 'ETTm4', 'ETTm5']:
        series = WindArgs(args.data)
    else:
        series = Args(args.data)
    try:
        spectral, category, Centure = series.train_fit()
        if Centure == True:
            args.cluster_enc_in = category.shape[0]
            print('load cluster centure')
        else:
            args.cluster_enc_in = category.centure_.shape[0]
            print('calculate cluster centure')
    except:
        args.cluster_enc_in = category.centure_.shape[0]
        print('calculate cluster centure')

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        if args.model == 'Windformer':
            exp.train(setting, spectral, category)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae = exp.test(setting, category=category)
            mses.append(mse)
            maes.append(mae)
        else:
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae = exp.test(setting)
            mses.append(mse)
            maes.append(mae)
        torch.cuda.empty_cache()

    print('average mse:{0:.3f}±{1:.3f}, mae:{2:.3f}±{3:.3f}'.format(np.mean(mses), np.std(mses),
                                                                    np.mean(maes),
                                                                    np.std(maes)))
    f = open("result_long_term_forecast.txt", 'a')
    f.write(u'average mse:{0:.3f}±{1:.3f}, mae:{2:.3f}±{3:.3f}'.format(np.mean(mses), np.std(mses),
                                                                       np.mean(maes),
                                                                       np.std(maes)))
    f.write('\n')
    f.write('\n')
    f.close()
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, category=category, test=1)
    torch.cuda.empty_cache()
