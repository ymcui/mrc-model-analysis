GS_BUCKET=gs://your-bucket
TPU_NAME=your-tpu-name
TPU_ZONE=your-tpu-zone
MODEL_OUTPUT_DIR=$GS_BUCKET/path-to-output-dir
for idx in {0..11};
do
python -u run_squad.py \
  --vocab_file=$GS_BUCKET/bert/cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$GS_BUCKET/bert/cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$GS_BUCKET/bert/cased_L-12_H-768_A-12/bert_model.ckpt \
  --do_train=False \
  --do_predict=True \
  --predict_file=./squad/dev-v1.1.json \
  --predict_batch_size=32 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --mask_layer=$idx \
  --mask_zone="q2" \
  --output_dir=$MODEL_OUTPUT_DIR \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --tpu_zone=$TPU_ZONE
done
