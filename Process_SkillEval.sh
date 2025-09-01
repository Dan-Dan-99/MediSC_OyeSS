#!/bin/bash

IMAGE_DIR="/input"
mv /output/gt/* /TrackEval/data/gt

process_image()
{
    local image_file=$1

    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_hands_plz_copy.py /MediSC_OyeSS/configs/best_coco_bbox_mAP_epoch_100_b16.pth /MediSC_OyeSS/configs/hand_pose_m.py /MediSC_OyeSS/configs/hand_pose_best_coco_AP_epoch_190.pth --input "$image_file" --det-cat-id 0 --save-predictions --output-root /MediSC_OyeSS/results/0_left_hand
    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_hands_plz_copy.py /MediSC_OyeSS/configs/best_coco_bbox_mAP_epoch_100_b16.pth /MediSC_OyeSS/configs/hand_pose_m.py /MediSC_OyeSS/configs/hand_pose_best_coco_AP_epoch_190.pth --input "$image_file" --det-cat-id 1 --save-predictions --output-root /MediSC_OyeSS/results/1_right_hand
    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_tools.py /MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /MediSC_OyeSS/configs/tool_pose.py /MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 0 --save-predictions --output-root /MediSC_OyeSS/results/2_scissors
    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_tools.py /MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /MediSC_OyeSS/configs/tool_pose.py /MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 1 --save-predictions --output-root /MediSC_OyeSS/results/3_tweezers
    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_tools.py /MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /MediSC_OyeSS/configs/tool_pose.py /MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 2 --save-predictions --output-root /MediSC_OyeSS/results/4_needle_holder
    python /MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /MediSC_OyeSS/configs/Co-DETR_tools.py /MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /MediSC_OyeSS/configs/tool_pose.py /MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 3 --save-predictions --output-root /MediSC_OyeSS/results/5_needle
}

for image_file in "$IMAGE_DIR"/*.png; do
    if [ -f "$image_file" ]; then
        process_image "$image_file"
    fi
done

python json_to_mot.py

python /TrackEval/scripts/run_mot_challenge_kp.py

mv /TrackEval/data/out/* /output
