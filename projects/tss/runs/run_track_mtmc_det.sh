#!/bin/bash

# Full path of the current script
THIS_DET=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
ROOT_DIR=$(dirname "${THIS_DET}")
ROOT_DIR=$(dirname "${ROOT_DIR}")
ROOT_DIR=$(dirname "${ROOT_DIR}")
ROOT_DIR=$(dirname "${ROOT_DIR}")

# seqs1=(c041)
# img_size=1280
# conf_thres=0.1
# gpu_id=1
 # shellcheck disable=SC2068
# for seq in ${seqs1[@]}
# do
#     CUDA_VISIBLE_DEVICES=${gpu_id} python3 projects/tss/src/detectors/yolov5v4/detect.py  \
#         --name   ${seq}  \
# 		--output ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/  \
# 		--source ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/images/${seq}    \
# 		--weights ${ROOT_DIR}/models_zoo/pretrained/detector/yolov5x.pt  \
# 		--roi_region ${ROOT_DIR}/projects/tss/data/aic22_mtmc/dataset_a/${seq}/roi.jpg \
# 		--conf 0.1 \
# 		--agnostic  \
# 		--save-txt  \
# 		--save-conf  \
# 		--classes 2 5 7 \
# 		--img-size 1280 &
# done


seqs1=(c041 c043 c045)
img_size=1280
conf_thres=0.1
gpu_id=1
# shellcheck disable=SC2068
for seq in ${seqs1[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 projects/tss/src/detectors/ScaledYOLOv4/detect.py  \
        --name   ${seq}  \
		--output ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/  \
		--source ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/images/${seq}  \
		--weights ${ROOT_DIR}/models_zoo/pretrained/scaledyolov4/yolov4-p7.pt  \
		--roi_region ${ROOT_DIR}/projects/tss/data/aic22_mtmc/dataset_a/${seq}/roi.jpg \
		--conf ${conf_thres}  \
		--agnostic  \
		--save-txt  \
		--save-conf  \
		--classes 2 5 7 \
		--img-size ${img_size} &
done

seqs2=(c042 c044 c046)
gpu_id=0
# shellcheck disable=SC2068
for seq in ${seqs2[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 projects/tss/src/detectors/ScaledYOLOv4/detect.py \
        --name   ${seq}  \
		--output ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/  \
		--source ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/images/${seq}  \
		--weights ${ROOT_DIR}/models_zoo/pretrained/scaledyolov4/yolov4-p7.pt  \
		--roi_region ${ROOT_DIR}/projects/tss/data/aic22_mtmc/dataset_a/${seq}/roi.jpg \
		--conf ${conf_thres}  \
		--agnostic  \
		--save-txt  \
		--save-conf  \
		--classes 2 5 7 \
		--img-size ${img_size} &
done

#gpu_id=3
#seq=c045
#CUDA_VISIBLE_DEVICES=${gpu_id} python3 projects/tss/src/detectors/yolov5v4/detect.py  \
#        --name   ${seq}  \
#		--output ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/  \
#		--source ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/images/${seq}  \
#		--weights ${ROOT_DIR}/models_zoo/pretrained/detector/yolov5x.pt  \
#		--roi_region ${ROOT_DIR}/projects/tss/data/aic22_mtmc/dataset_a/${seq}/roi.jpg \
#		--conf ${conf_thres}  \
#		--agnostic  \
#		--save-txt  \
#		--save-conf  \
#		--classes 2 5 7 \
#		--img-size ${img_size} &
#
#gpu_id=2
#seq=c046
#CUDA_VISIBLE_DEVICES=${gpu_id} python3 projects/tss/src/detectors/yolov5v4/detect.py  \
#        --name   ${seq}  \
#		--output ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/  \
#		--source ${ROOT_DIR}/projects/tss/data/aic22_mtmc/outputs/images/${seq}  \
#		--weights ${ROOT_DIR}/models_zoo/pretrained/detector/yolov5x.pt  \
#		--roi_region ${ROOT_DIR}/projects/tss/data/aic22_mtmc/dataset_a/${seq}/roi.jpg \
#		--conf ${conf_thres}  \
#		--agnostic  \
#		--save-txt  \
#		--save-conf  \
#		--classes 2 5 7 \
#		--img-size ${img_size}

wait