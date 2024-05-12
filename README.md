# SDTP(ESWA 2024)

SDTP: Series decomposition Transformer with period-correlation for stock market
index prediction


Stock price forecasting has been always a difficult and crucial undertaking in the field of finance. Enlighted by the classic time series analysis and stochastic process theory, we propose the SDTP as a safe and reliable stock prediction model [[paper](https://www.sciencedirect.com/science/article/pii/S0957417423019267?via%3Dihub)]. **SDTP goes beyond the baseline models and performs well in most stock markets.**


## SDTP model

**1.  Series decomposition layer**

It is difficult to
learn the complex temporal patterns directly in the financial series, and
we are inspired by the work of Cleveland (1990), Taylor and Letham(2018), Wu, Xu, Wang, and Long (2021), which decomposes the series
into the trend and seasonal parts. 

<p align="center">
<img src=".\pic\SDTP.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of SDTP.
</p>

**2. Period-correlation mechanism**

Observing that the nodes in the corresponding
positions in the period-based sub-series usually have similar properties
to each other (Xu, Li, Qian, & Wang, 2022), we are inspired by Wu,
Xu, et al. (2021) to use period-correlation instead of traditional self-attention, which can detect the periodic relationship between the series
more reasonably.

<p align="center">
<img src=".\pic\Auto-Correlation.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Period-correlation mechanism.
</p>

## Requirements

---
* Python 3.8
* pandas==1.5.2
* torch==1.12.1
* numpy==1.23.4
* matplotlib==3.7.0
* tensorflow==2.11.0
* transformers==4.26.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Usage

Commands for training and testing the model on Dataset SH:

```bash
# SH
python -u run.py --model SDTP --data custom  --root_path ./ETT-small/    --data_path  price_SH.csv --freq d --enc_in 8 --dec_in 8 --c_out 8 --target closeIndex --train_epochs 100 --batch_size 64 --features MS --itr 3 --seq_len 5 --label_len 5 --pred_len 1 --dropout 0.05 --learning_rate 0.001 --patience 50 --moving_avg 5 --lradj type1 --e_layers 5 --d_layers 3 --d_ff 1024 --d_model 512

```

More parameter information please refer to `run.py`.

We provide a more detailed and complete command description for training and testing the model:

```python
python -u SDTP.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--s_layers <s_layers> --d_ff <d_ff> --factor <factor> --padding <padding>
--distil --dropout <dropout> --attn <attn> --embed <embed> --activation <activation>
--output_attention --do_predict --mix --cols <cols> --itr <itr>
--num_workers <num_workers> --train_epochs <train_epochs>
--batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_amp --inverse --use_gpu <use_gpu> --gpu <gpu> --use_multi_gpu --devices <devices>
```

The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter                                                                                                                                                                                                                 |
| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model | The model of experiment. This can be set to `SDTP`, `informer`, `TimesNet`                                                                                                                                                               |
| data           | The dataset name                                                                                                                                                                                                                         |
| root_path      | The root path of the data file (defaults to `./data/ETT/`)                                                                                                                                                                               |
| data_path      | The data file name (defaults to `ETTh1.csv`)                                                                                                                                                                                             |
| features       | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate)                                                 |
| target         | Target feature in S or MS task (defaults to `OT`)                                                                                                                                                                                        |
| freq           | Freq for time features encoding (defaults to `h`). This can be set to `s`,`t`,`h`,`d`,`b`,`w`,`m` (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h |
| checkpoints    | Location of model checkpoints (defaults to `./checkpoints/`)                                                                                                                                                                             |
| seq_len | Input sequence length of Informer encoder (defaults to 5)                                                                                                                                                                                |
| label_len | Start token length of Informer decoder (defaults to 5)                                                                                                                                                                                   |
| pred_len | Prediction sequence length (defaults to 1)                                                                                                                                                                                               |
| enc_in | Encoder input size (defaults to 7)                                                                                                                                                                                                       |
| dec_in | Decoder input size (defaults to 7)                                                                                                                                                                                                       |
| c_out | Output size (defaults to 7)                                                                                                                                                                                                              |
| d_model | Dimension of model (defaults to 512)                                                                                                                                                                                                     |
| n_heads | Num of heads (defaults to 8)                                                                                                                                                                                                             |
| e_layers | Num of encoder layers (defaults to 2)                                                                                                                                                                                                    |
| d_layers | Num of decoder layers (defaults to 1)                                                                                                                                                                                                    |
| s_layers | Num of stack encoder layers (defaults to `3,2,1`)                                                                                                                                                                                        |
| d_ff | Dimension of fcn (defaults to 2048)                                                                                                                                                                                                      |
| factor | Probsparse attn factor (defaults to 5)                                                                                                                                                                                                   |
| padding | Padding type(defaults to 0).                                                                                                                                                                                                             |
| distil | Whether to use distilling in encoder, using this argument means not using distilling (defaults to `True`)                                                                                                                                |
| dropout | The probability of dropout (defaults to 0.05)                                                                                                                                                                                            |
| attn | Attention used in encoder (defaults to `prob`). This can be set to `prob` (informer), `full` (transformer)                                                                                                                               |
| embed | Time features encoding (defaults to `timeF`). This can be set to `timeF`, `fixed`, `learned`                                                                                                                                             |
| activation | Activation function (defaults to `gelu`)                                                                                                                                                                                                 |
| output_attention | Whether to output attention in encoder, using this argument means outputing attention (defaults to `False`)                                                                                                                              |
| do_predict | Whether to predict unseen future data, using this argument means making predictions (defaults to `False`)                                                                                                                                |
| mix | Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to `True`)                                                                                                               |
| cols | Certain cols from the data files as the input features                                                                                                                                                                                   |
| num_workers | The num_works of Data loader (defaults to 0)                                                                                                                                                                                             |
| itr | Experiments times (defaults to 2)                                                                                                                                                                                                        |
| train_epochs | Train epochs (defaults to 6)                                                                                                                                                                                                             |
| batch_size | The batch size of training input data (defaults to 32)                                                                                                                                                                                   |
| patience | Early stopping patience (defaults to 3)                                                                                                                                                                                                  |
| learning_rate | Optimizer learning rate (defaults to 0.0001)                                                                                                                                                                                             |
| des | Experiment description (defaults to `test`)                                                                                                                                                                                              |
| loss | Loss function (defaults to `mae`)                                                                                                                                                                                                        |
| lradj | Ways to adjust the learning rate (defaults to `type1`)                                                                                                                                                                                   |
| use_amp | Whether to use automatic mixed precision training, using this argument means using amp (defaults to `False`)                                                                                                                             |
| inverse | Whether to inverse output data, using this argument means inversing output data (defaults to `False`)                                                                                                                                    |
| use_gpu | Whether to use gpu (defaults to `True`)                                                                                                                                                                                                  |
| gpu | The gpu no, used for training and inference (defaults to 0)                                                                                                                                                                              |
| use_multi_gpu | Whether to use multiple gpus, using this argument means using mulitple gpus (defaults to `False`)                                                                                                                                        |
| devices | Device ids of multile gpus (defaults to `0,1,2,3`)                                                                                                                                                                                       |

**The above parameters are for reference only, the actual values please refer to the code.**
## Data
The required data files should be put into `/ETT-small/` folder. 
we select three well-known stock indices, the Shanghai
Composite Index (SH) from July 1, 1991 to June 30, 2020, and the
Shenzhen Component Index (SZ) and the Hang Seng Index (HSI) from
July 31, 1991 to May 31, 2022.   
To further demonstrate the feasibility of our proposed SDTP model
for the stock price prediction, we conduct the experiments with the
Reservoir Computing (RC) (Wang et al., 2021) model in eight different stock indices,
including DJI, the Nasdaq Composite Index, the SSE Index, the FTSE
100 Index, the Nikkei 225 Index, the NYSE Composite Index, the CAC40
index, and the S&P500 Index.
<p align="center">
<img src=".\pic\data1.png" height = "" alt="" align=center />
</p>
<p align="center">
<img src=".\pic\data2.png" height = "" alt="" align=center />
</p>

##  Main Results

We compare our model with benchmarks in popular stock indexes, including SH, SZ, etc. **Generally, SDTP achieves SOTA** over previous baselines.

<p align="center">
<img src=".\pic\exp1.png" height = "" alt="" align=center />
</p>
<p align="center">
<img src=".\pic\exp2.png" height = "" alt="" align=center />
</p>

## Baselines

We will keep adding series forecasting models to expand this repo:

- [x] TimesNet
- [x] Informer
- [x] Transformer
- [x] Reformer
- [x] CNN-BiLSTM-AM
- [x] Reservoir Computing

The baseline code is in `/baseline` and `/models`
## Citation

If you find this repo useful, please cite our paper. 

```
@article{TAO2024121424,
title = {Series decomposition Transformer with period-correlation for stock market index prediction},
journal = {Expert Systems with Applications},
volume = {237},
pages = {121424},
year = {2024}
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

