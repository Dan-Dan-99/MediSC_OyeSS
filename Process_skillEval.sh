#!/bin/bash

IMAGE_DIR="/input"
mv /output/gt/* /workspace/TrackEval/data/gt

process_image()
{
    local image_file=$1

    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_hands_plz_copy.py /workspace/MediSC_OyeSS/configs/best_coco_bbox_mAP_epoch_100_b16.pth /workspace/MediSC_OyeSS/configs/hand_pose_m.py /workspace/MediSC_OyeSS/configs/hand_pose_best_coco_AP_epoch_190.pth --input "$image_file" --det-cat-id 0 --save-predictions --output-root /workspace/MediSC_OyeSS/results/0_left_hand
    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_hands_plz_copy.py /workspace/MediSC_OyeSS/configs/best_coco_bbox_mAP_epoch_100_b16.pth /workspace/MediSC_OyeSS/configs/hand_pose_m.py /workspace/MediSC_OyeSS/configs/hand_pose_best_coco_AP_epoch_190.pth --input "$image_file" --det-cat-id 1 --save-predictions --output-root /workspace/MediSC_OyeSS/results/1_right_hand
    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_tools.py /workspace/MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /workspace/MediSC_OyeSS/configs/tool_pose.py /workspace/MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 0 --save-predictions --output-root /workspace/MediSC_OyeSS/results/2_scissors
    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_tools.py /workspace/MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /workspace/MediSC_OyeSS/configs/tool_pose.py /workspace/MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 1 --save-predictions --output-root /workspace/MediSC_OyeSS/results/3_tweezers
    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_tools.py /workspace/MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /workspace/MediSC_OyeSS/configs/tool_pose.py /workspace/MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 2 --save-predictions --output-root /workspace/MediSC_OyeSS/results/4_needle_holder
    python /workspace/MediSC_OyeSS/demo/topdown_demo_with_mmdet.py /workspace/MediSC_OyeSS/configs/Co-DETR_tools.py /workspace/MediSC_OyeSS/configs/tool_best_coco_bbox_mAP_epoch_170.pth /workspace/MediSC_OyeSS/configs/tool_pose.py /workspace/MediSC_OyeSS/configs/tool_best_epoch_250.pth --input "$image_file" --det-cat-id 3 --save-predictions --output-root /workspace/MediSC_OyeSS/results/5_needle
}

for image_file in "$IMAGE_DIR"/*.png; do
    if [ -f "$image_file" ]; then
        process_image "$image_file"
    fi
done

python /workspace/MediSC_OyeSS/json_to_mot.py

python /workspace/TrackEval/scripts/run_mot_challenge_kp.py

mv /workspace/TrackEval/data/out/* /output
