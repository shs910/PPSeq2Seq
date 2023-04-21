# 微调一个复述小模型
Transformers_path=/home/hssun/sunhuashan/transformers

Lunach_File_path=/home/hssun/sunhuashan/PPSeq2Seq/fine_tuning/run_seq2seq_task_train.py
Model_Name=ramsrigouthamg/t5-large-paraphraser-diverse-high-quality
WANDB_PROJECT_NAME=fine_tune_paraphrase_model
# 每次不同的实验时，可以修改run_name参数简单描述该实验情况）
# iwslt_only,iwslt_news,iwslt_transfered_news
for RUN_NAME in _12w
do 
    Train_data_path=/home/hssun/sunhuashan/data/news2oral/train/train_no_instruction$RUN_NAME.json
    Valid_data_path=/home/hssun/sunhuashan/data/news2oral/train/valid_no_instruction.json
    Output_dir=/home/hssun/sunhuashan/PPSeq2Seq/fine_tuning/$RUN_NAME
    # train with trainer
    # 使用MBART官方给的FT参数
    # https://github.com/facebookresearch/fairseq/tree/main/examples/mbart
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" $Lunach_File_path \
        --model_name_or_path $Model_Name \
        --do_train \
        --do_eval \
        --task StyleTransfer \
        --train_file  $Train_data_path\
        --validation_file  $Valid_data_path\
        --source_lang src_XX \
        --target_lang tgt_XX \
        --tgt_prefix "paraphrasedoutput: " \
        --output_dir $Output_dir \
        --per_device_train_batch_size 3 \
        --per_device_eval_batch_size 6 \
        --gradient_accumulation_steps 4 \
        --label_smoothing_factor 0.2 \
        --lr_scheduler_type  cosine \
        --learning_rate  3e-05 \
        --warmup_ratio 0.03 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-06 \
        --predict_with_generate \
        --evaluation_strategy steps \
        --eval_steps  50 \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --report_to wandb \
        --project_name $WANDB_PROJECT_NAME \
        --num_train_epochs 2 \
        --run_name $RUN_NAME \
        --generation_max_length 128 
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap "MBartEncoderLayer MBartDecoderLayer" \
        # --resume_from_checkpoint=/home/hssun/sunhuashan/MachineTranslation/output/iwslt_only/checkpoint-1000
        # --overwrite_output_dir \
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 
done
# predict_with_generate是对tes数据
