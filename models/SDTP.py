import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # print(x_enc.shape,x_mark_enc.shape,x_dec.shape,x_mark_dec.shape)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

def plot_model(model, inputs):
    """
    inputs:为输入参数种类,
    """
    import onnx
    import onnx.utils
    import onnx.version_converter
    # 导出模型为onnx格式
    torch.onnx.export(
        model=model,
        args=inputs,
        f='model1.onnx',
        # input_names=['input1', 'input2', 'input3', 'input4'],
        # output_names=['model_args'],
        verbose=True,
        export_params=True,
        opset_version=13,
    )
    model_file = 'model1.onnx'
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)  # 增加维度信息

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
#
#     # basic config
#
#     # data loader
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#
#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#
#     # model define
#     parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
#     parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
#     parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=7, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false',
#                         help='whether to use distilling in encoder, using this argument means not using distilling',
#                         default=True)
#     parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
#     parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
#
#     # optimization
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#
#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=3, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#     args = parser.parse_args(['--moving_avg', '5',
#                               '--e_layers', '5', '--d_layers', '3', '--d_ff', '1024', '--d_model', '512'])
#     Auto = Model(configs=args)
#     input1 = torch.randn(64, 5, 7)
#     input2 = torch.randn(64, 5, 3)
#     input3 = torch.randn(64, 6, 7)
#     input4 = torch.randn(64, 6, 3)
#     inputs = (input1, input2, input3, input4)
#     plot_model(model=Auto, inputs=inputs)