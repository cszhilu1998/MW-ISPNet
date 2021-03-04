
LOG=./log/'train_track1_'+`date +%Y-%m-%d-%H-%M-%S`.txt
python train_track1.py --dataroot ./data/Zurich-RAW-to-DSLR/  \
                       --save_model_path ./ckpt/Track1/track1_train/ | tee $LOG
