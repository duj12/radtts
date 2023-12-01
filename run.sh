#!/usr/bin/env bash

stage=$1

#   训练单一数据集(直播)的radtts模型
if [ $stage -eq 1 ]; then
  train_decoder=true
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=2 \
      nohup python train_xmov.py \
      -c configs/config_xmov_radtts.json \
      -p train_config.output_directory=exp/radtts_xmov_s1
      model_config.include_modules=decatn >> train_radtts_xmov_s1.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络
    CUDA_VISIBLE_DEVICES=3 \
      nohup python train_xmov.py \
      -c configs/config_xmov_radtts.json \
      -p train_config.output_directory=exp/radtts_xmov_s2 \
      train_config.warmstart_checkpoint_path=exp/radtts_xmov_s1/model_200000 \
      model_config.include_modules=decatndpm >> train_radtts_xmov_s2.log 2>&1 &
  fi
fi

# 训练单一数据集(直播)的radtts++模型
if [ $stage -eq 2 ]; then
  train_decoder=true
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=4 \
      nohup python train_xmov.py \
      -c configs/config_xmov_decoder.json \
      -p train_config.output_directory=exp/radtts++_xmov_s1 >> train_radtts++xmov_s1.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络,f0和energy等属性预测网络
    CUDA_VISIBLE_DEVICES=5 \
      nohup python train_xmov.py \
      -c configs/config_xmov_agap.json \
      -p train_config.output_directory=exp/radtts++_xmov_s2_agap \
      train_config.warmstart_checkpoint_path=exp/radtts++_xmov_s1/model_200000 >> train_radtts++xmov_s2.log 2>&1 &
  fi
fi

# 训练全部数据的radtts模型
if [ $stage -eq 3 ]; then
  train_decoder=false
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=0 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_radtts.json \
      -p train_config.output_directory=exp/radtts_xmov_alldata_s1 \
      model_config.include_modules=decatn >> train_radtts_xmov_alldata_s1.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络
    CUDA_VISIBLE_DEVICES=0 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_radtts.json \
      -p train_config.output_directory=exp/radtts_xmov_alldata_s2 \
      train_config.warmstart_checkpoint_path=exp/radtts_xmov_alldata_s1/model_480000 \
      model_config.include_modules=decatndpm >> train_radtts_xmov_alldata_s2.log 2>&1 &
  fi
fi

# 训练全部数据集的radtts++模型
if [ $stage -eq 4 ]; then
  train_decoder=false
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=1 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_decoder.json \
      -p train_config.output_directory=exp/radtts++_xmov_alldata_s1 >> train_radtts++xmov_alldata_s1.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络,f0和energy等属性预测网络
    CUDA_VISIBLE_DEVICES=1 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_agap.json \
      -p train_config.output_directory=exp/radtts++_xmov_alldata_s2_agap \
      train_config.warmstart_checkpoint_path=exp/radtts++_xmov_alldata_s1/model_480000 >> train_radtts++xmov_alldata_s2.log 2>&1 &
  fi
fi

# 训练全部数据的radtts模型, 对齐不用到spk_emb, 为了处理未知说话人，只根据text_emb对齐
if [ $stage -eq 5 ]; then
  train_decoder=false
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=6 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_radtts_alnwospk.json \
      -p train_config.output_directory=exp/radtts_xmov_alldata_s1_alnwospk \
      model_config.include_modules=decatn >> train_radtts_xmov_alldata_s1_alnwospk.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络
    CUDA_VISIBLE_DEVICES=6 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_radtts_alnwospk.json \
      -p train_config.output_directory=exp/radtts_xmov_alldata_s2_alnwospk \
      train_config.warmstart_checkpoint_path=exp/radtts_xmov_alldata_s1_alnwospk/model_500000 \
      model_config.include_modules=decatndpm >> train_radtts_xmov_alldata_s2_alnwospk.log 2>&1 &
  fi
fi

# 训练全部数据集的radtts++模型, 对齐不用到spk_emb, 为了处理未知说话人，只根据text_emb对齐
if [ $stage -eq 6 ]; then
  train_decoder=false
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=7 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_decoder_alnwospk.json \
      -p train_config.output_directory=exp/radtts++_xmov_alldata_s1_alnwospk >> train_radtts++xmov_alldata_s1_alnwospk.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络,f0和energy等属性预测网络
    CUDA_VISIBLE_DEVICES=7 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_agap_alnwospk.json \
      -p train_config.output_directory=exp/radtts++_xmov_alldata_s2_agap_alnwospk \
      train_config.warmstart_checkpoint_path=exp/radtts++_xmov_alldata_s1_alnwospk/model_500000 >> train_radtts++xmov_alldata_s2_alnwospk.log 2>&1 &
  fi
fi

# 训练全部数据的radtts模型, 对齐不用到spk_emb, 为了处理未知说话人，只根据text_emb对齐
if [ $stage -eq 7 ]; then
  train_decoder=true
  if $train_decoder; then
    # 第一阶段，先只训练声学模型网络层, 至少200000步
    CUDA_VISIBLE_DEVICES=0 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_styletts_16k.json \
      -p train_config.output_directory=exp/styletts_xmov_alldata_s1 \
      model_config.include_modules=decatn >> train_styletts_xmov_alldata_s1.log 2>&1 &
  else
    # 第二阶段，基于第一阶段已经训练差不多的网络，再训练时长预测网络
    CUDA_VISIBLE_DEVICES=0 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_styletts_16k.json \
      -p train_config.output_directory=exp/styletts_xmov_alldata_s2 \
      train_config.warmstart_checkpoint_path=exp/styletts_xmov_alldata_s1/model_500000 \
      model_config.include_modules=decatndpmgst >> train_styletts_xmov_alldata_s2.log 2>&1 &
  fi
fi

# 训练全部数据的radtts模型, 对齐不用到spk_emb, 为了处理未知说话人，只根据text_emb对齐
if [ $stage -eq 8 ]; then
    # 直接尝试从头训encoder decoder gst 和 dp
    CUDA_VISIBLE_DEVICES=1 \
      nohup python train_xmov.py \
      -c configs/config_xmov_alldata_styletts_16k.json \
      -p train_config.output_directory=exp/styletts_xmov_alldata \
      model_config.include_modules=decatndpmgst >> train_styletts_xmov_alldata.log 2>&1 &
fi