# base_57.8.pth
python eval_esp.py --method student_esp_d \
--dataset camvid_light --data_list /datasets/segment_dataset/CamVid11/test.txt \
--data_dir /datasets/segment_dataset/CamVid11 \
--num_classes 11 \
--restore-from ./checkpoint/Camvid/ESP/base_57.8.pth \
--store-output False

# kd_59.8.pth
python eval_esp.py --method student_esp_d \
--dataset camvid_light --data_list /datasets/segment_dataset/CamVid11/test.txt \
--data_dir /datasets/segment_dataset/CamVid11 \
--num_classes 11 \
--restore-from ./checkpoint/Camvid/ESP/kd_59.8.pth \
--store-output False