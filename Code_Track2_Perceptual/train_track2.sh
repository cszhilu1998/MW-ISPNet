
LOG=./log/'train_track2_'+`date +%Y-%m-%d-%H-%M-%S`.txt
python train_track2.py --dataroot ./data/Zurich-RAW-to-DSLR/  \
                       --save_model_path ./ckpt/Track2/track2_train/ | tee $LOG
