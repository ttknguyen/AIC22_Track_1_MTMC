#!/bin/bash

# Add python path for MLKit
export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")

# NOTE: IMAGE EXTRACTION
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c041.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c042.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c043.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c044.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c045.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --images_extraction \
--config "c046.yaml"
wait

# NOTE: DETECTION
bash "${DIR_CURRENT}/run_track_mtmc_det.sh"
wait

# NOTE: FEATURE EXTRACTION
bash "${DIR_CURRENT}/run_track_mtmc_reid.sh"

# NOTE: MERGE ALL FEATURES
python3 projects/tss/runs/main_aic21_track_mtmc.py  --dataset "aic22_mtmc" \
    --config "default.yaml" --feature_merge --config_feat_merge "reid_merge.yaml"
wait

# NOTE: RUN DETECTIONS FILTER
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c041.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c042.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c043.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c044.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c045.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --detection_filter \
--config "c046.yaml"
wait

# NOTE: RUN TRACKING
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c041.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c042.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c043.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c044.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c045.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --tracking --use_dets_feat_filter \
   --config "c046.yaml"
wait

# NOTE: MATCHING ZONE
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c041.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c042.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c043.yaml"
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c044.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c045.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_zone --use_mots_feat_raw \
   --config "c046.yaml"
wait

# NOTE: SUB-CLUSTER
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --matching_scene \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

# NOTE: WRITE RESULT
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --write_result \
    --config "default.yaml" --config_feat_merge "reid_merge.yaml"
wait

END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
