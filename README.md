# USS: Universal Sound Separation Framework

通用语音分离模型的统一训练与推理框架。

## 目录结构

```
USS/
├── train.py                        # 训练入口
├── inference.py                    # 推理入口
├── configs/
│   └── flowsep/
│       └── flowsep.yaml
├── core/
│   ├── audio/
│   │   ├── audio_processing.py     # 动态范围压缩、Griffin-Lim
│   │   ├── stft.py                 # STFT / TacotronSTFT
│   │   └── tools.py               # get_mel_from_wav
│   ├── data/
│   │   ├── dataset.py              # AudioDataset
│   │   └── big_vgan_mel.py         # BigVGAN mel 提取
│   ├── model.py                    # get_vocoder / vocoder_infer
│   └── tools.py                    # load_json / get_restore_step
├── backbones/
│   ├── latent_diffusion/
│   │   ├── models/
│   │   │   └── ddpm_flow.py        # LatentDiffusion (FlowSep 主模型)
│   │   ├── modules/
│   │   │   ├── attention.py
│   │   │   ├── ema.py
│   │   │   ├── audiomae/           # AudioMAE 编码器
│   │   │   ├── diffusionmodules/
│   │   │   │   ├── model.py        # Encoder / Decoder
│   │   │   │   ├── openaimodel.py  # UNetModel
│   │   │   │   └── util.py
│   │   │   ├── distributions/
│   │   │   ├── encoders/
│   │   │   │   └── modules.py      # FlanT5HiddenState
│   │   │   ├── losses/
│   │   │   │   ├── contperceptual.py
│   │   │   │   ├── waveform_contperceptual.py
│   │   │   │   ├── waveform_contperceptual_panns.py
│   │   │   │   └── panns_distance/
│   │   │   └── phoneme_encoder/
│   │   └── util.py                 # instantiate_from_config
│   └── latent_encoder/
│       ├── autoencoder.py          # AutoencoderKL (VAE)
│       └── wavedecoder/
├── bigvgan/                        # BigVGAN 声码器
│   ├── model.py
│   ├── config.json                 # (需用户放置)
│   └── g_01000000                  # (需用户放置)
└── models/
```

## 模型

- **FlowSep**: 基于 Rectified Flow 的语言查询语音分离

## 使用

训练:
```
python train.py -c configs/flowsep/flowsep.yaml
```

推理:
```
python inference.py -c configs/flowsep/flowsep.yaml -t "text query" -a "audio.wav" -l checkpoint.ckpt
```

## 预训练权重

BigVGAN 声码器权重需放置在 `USS/bigvgan/` 目录下（或通过 `BIGVGAN_PATH` 环境变量指定）:
- `bigvgan/config.json`
- `bigvgan/g_01000000`

VAE 权重路径在配置文件 `first_stage_config.params.reload_from_ckpt` 中指定。

## 依赖

参考 FlowSep/AudioLDM 环境配置，需安装: taming-transformers, pytorch-lightning, transformers 等。
