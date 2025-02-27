# unshard脚本: 注意output_dir必须要以unsharded结尾才能正常重新加载
python3 scripts/unshard.py \
  --input_dir /opt/tiger/olmo/olmo2_exps/debug_olmo2/step100 \
  --output_dir /opt/tiger/olmo/olmo2_exps/debug_olmo2/step100-unsharded \
  --model-only


# olmo2_to_hf脚本: 注意如果修改了模型结构需要更改modeling_olmo.py
# 1. transform model_ckpt (NOTE: pip3 install --upgrade transformers==4.47.0)
# 2. 恢复tokenizer: hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yourname/corpus/olmo2_related/olmo2_tokenizer_files
python3 scripts/convert_olmo2_to_hf.py \
  --input_dir /opt/tiger/olmo/olmo2_exps/debug_olmo2/step100-unsharded \
  --output_dir /opt/tiger/olmo/olmo2_exps/debug_olmo2/step100-unsharded-hf \
  --no_fix_eos_token_id \
  --no_tokenizer

cp -a /mnt/bn/mount_nas/datasets/olmo2_related/olmo2_tokenizer_files/* /opt/tiger/olmo/olmo2_exps/debug_olmo2/step100-unsharded-hf/