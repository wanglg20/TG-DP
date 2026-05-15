vggsound_label_csv='src/data_info/class_labels_indices_vgg.csv'
model_names='TG-DP'
model_types='sync_pretrain_registers_cls_4s'




# modify the path to your own path
vggsound_info_path='/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/data_info/VGG/vgg_partition_test_5_per_class.json'
model_paths='/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/exp/cvpr_draft/best_audio_model.pth'



python src/retrieval.py --nums_samples 1600 --directions audio video --strategy diagonal_mean \
 --vgg_data_path $vggsound_info_path --vgg_label_csv $vggsound_label_csv \
 --model_names $model_names --model_paths $model_paths --model_types $model_types