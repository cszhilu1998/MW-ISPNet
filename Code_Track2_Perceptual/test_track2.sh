
LOG=./log/'test_track2_'+`date +%Y-%m-%d-%H-%M-%S`.txt
python test_track2.py --dataroot /data/dataset/Zurich-RAW-to-DSLR/test/huawei_raw  \
                      --data_size 1204 \
                      --save_img_path ./results/Track2/test/  \
                      --fullres False | tee $LOG
