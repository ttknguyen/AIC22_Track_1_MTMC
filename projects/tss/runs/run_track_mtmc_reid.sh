#!/bin/bash

# NOTE: FEATURE EXTRACTION - 1
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c041_1.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c042_1.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c043_1.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c044_1.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c045_1.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c046_1.yaml"
wait

# NOTE: FEATURE EXTRACTION - 2
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c041_2.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c042_2.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c043_2.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c044_2.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c045_2.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c046_2.yaml"
wait

# NOTE: FEATURE EXTRACTION - 3
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c041_3.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c042_3.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c043_3.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c044_3.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c045_3.yaml" &
python3 projects/tss/runs/main_aic21_track_mtmc.py --dataset "aic22_mtmc"  --feature_extraction \
--config "reid/c046_3.yaml"
wait

