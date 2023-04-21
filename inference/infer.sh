# nohup python /home/hssun/sunhuashan/PPSeq2Seq/inference/infer.py --not_log --loss_not_divded --file_title t5_1000_ --device_id 1 > infer_t5.log &;
# nohup python /home/hssun/sunhuashan/PPSeq2Seq/inference/infer.py --not_log --file_title t5_1000_loss_divid --batch_size 20 --device_id 1 > infer_t5.divide.log &;
# 2063
nohup python /home/hssun/sunhuashan/PPSeq2Seq/inference/infer.py \
    --pretrained_model tuner007/pegasus_paraphrase \
    --model_type pegasus \
    --classifier_model_path /home/hssun/sunhuashan/PPSeq2Seq/results/Pegasus/oral_classifier_head_best.pt \
    --not_log --file_title pegesus_1000_ --device_id 1 > infer_pegasus.log &;
# 3976
nohup python ./infer.py --not_log --file_title t5_large_ --batch_size 30 --loss_type 1 --gamma 1.1 > infer_t5_large.log &