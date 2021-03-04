
LOG=./log/'test_track1_'+`date +%Y-%m-%d-%H-%M-%S`.txt
python test_track1.py --dataroot /data/dataset/Zurich-RAW-to-DSLR/test/huawei_raw \
                      --data_size 1204 --pre_denoising False \
                      --save_img_path ./results/Track1/track1/ \
                      --fullres False | tee $LOG
