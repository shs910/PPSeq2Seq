export http_proxy="http:/127.0.0.1:7890"
# 微调一个复述小模型
Transformers_path=/home/hssun/sunhuashan/transformers

Lunach_File_path=/home/hssun/sunhuashan/PPSeq2Seq/fine_tuning/run_seq2seq_task_infer.py
Model_Name=/home/hssun/sunhuashan/PPSeq2Seq/fine_tuning/_5w
# WANDB_PROJECT_NAME=fine_tune_paraphrase_model
# 每次不同的实验时，可以修改run_name参数简单描述该实验情况）
# iwslt_only,iwslt_news,iwslt_transfered_news
# RUN_NAME=test1

# Train_data_path=/home/hssun/sunhuashan/data/news2oral/train/train_no_instruction.json
# Valid_data_path=/home/hssun/sunhuashan/data/news2oral/train/valid_no_instruction.json
Test_data_path=/home/hssun/sunhuashan/data/news2oral/train/test50.json
Output_dir=/home/hssun/sunhuashan/code/TST/data/

# train with trainer
# 使用MBART官方给的FT参数
# https://github.com/facebookresearch/fairseq/tree/main/examples/mbart

CUDA_VISIBLE_DEVICES=4 \
    python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost"  $Lunach_File_path \
    --do_predict \
    --num_beams 5 \
    --test_file $Test_data_path \
    --model_name_or_path $Model_Name \
    --task StyleTransfer \
    --source_lang src_XX \
    --target_lang tgt_XX \
    --output_dir $Output_dir \
    --per_device_eval_batch_size 2 \
    --predict_with_generate
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap "MBartEncoderLayer MBartDecoderLayer" \
    # --resume_from_checkpoint=/home/hssun/sunhuashan/MachineTranslation/output/iwslt_only/checkpoint-1000
    # --overwrite_output_dir \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 

# predict_with_generate是对tes数据