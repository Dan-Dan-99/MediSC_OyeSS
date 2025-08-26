import pandas as pd
import numpy as np
import os
import re
from numpy import array, float32

# eval()이 numpy 자료 구조를 이해하는 데 필요한 import
def parse_img_path(img_path):
    """이미지 경로에서 이미지 ID와 프레임 번호를 추출합니다."""
    basename = os.path.basename(img_path)
    filename, _ = os.path.splitext(basename)
    match = re.match(r'([a-zA-Z0-9]+)_frame_(\d+)', filename)
    if match:
        image_id = match.group(1)
        frame = int(match.group(2))
        return image_id, frame
    parts = filename.split('_frame_')
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return filename, 0

def process_results(data):
    """처리된 딕셔너리 리스트를 기반으로 CSV 데이터를 구조화합니다."""
    results_by_image_id = {}
    for record in data:
        # Check if the record has the 'img_path' attribute
        if not hasattr(record, 'img_path'): continue
        
        img_path = record.img_path
        image_id, frame = parse_img_path(img_path)

        # Access attributes directly using dot notation
        if hasattr(record, 'pred_instances') and record.pred_instances:
            pred_instances = record.pred_instances
            
            # Access attributes within the 'pred_instances' object
            keypoints_all = pred_instances.keypoints
            scores_all = pred_instances.keypoint_scores

            if keypoints_all is None or scores_all is None: continue
            
            # Access the category_id attribute
            class_id_raw = record.category_id
            class_id = class_id_raw
            if isinstance(class_id_raw, np.ndarray):
                class_id = class_id_raw.item() if class_id_raw.size == 1 else class_id_raw

            num_instances = keypoints_all.shape[0]
            for i in range(num_instances):
                instance_class_id = class_id[i] if isinstance(class_id, np.ndarray) and class_id.size > 1 else class_id
                
                row = {
                    'Frame': frame,
                    'Track ID': instance_class_id,
                    'Class ID': instance_class_id,
                    'Bbox_x': -1,
                    'Bbox_y': -1,
                    'Bbox_w': -1,
                    'Bbox_h': -1,
                }
                
                keypoints = keypoints_all[i]
                scores = scores_all[i]
                for kp_idx in range(keypoints.shape[0]):
                    x, y = keypoints[kp_idx]
                    score = scores[kp_idx]
                    row[f'KP{kp_idx+1}_x'] = x
                    row[f'KP{kp_idx+1}_y'] = y
                    row[f'KP{kp_idx+1}_c'] = score
                
                if image_id not in results_by_image_id:
                    results_by_image_id[image_id] = []
                results_by_image_id[image_id].append(row)
    return results_by_image_id

def save_to_csv(results_by_image_id):
    """
    처리된 데이터를 이미지 ID별로 텍스트 파일에 저장합니다.
    각 행은 자체 키포인트 개수를 유지하며, 0으로 채우지 않습니다.
    """
    if not results_by_image_id:
        print("처리된 데이터가 없습니다. 파일이 생성되지 않습니다.")
        return

    for image_id, rows in results_by_image_id.items():
        if not rows: continue
        
        txt_filename = f"{image_id}_pred.txt"
        
        with open(txt_filename, 'w') as f:
            for row in rows:  # row는 딕셔너리입니다.
                kp_keys = [k for k in row.keys() if k.startswith('KP')]
                max_kp_num = 0
                if kp_keys:
                    max_kp_num = max(int(re.search(r'KP(\d+)_', key).group(1)) for key in kp_keys)

                line_parts = []
                static_cols = ['Frame', 'Track ID', 'Class ID', 'Bbox_x', 'Bbox_y', 'Bbox_w', 'Bbox_h']
                
                for col in static_cols:
                    value = row.get(col, -1)
                    line_parts.append(str(int(value)))

                for i in range(1, max_kp_num + 1):
                    x = row.get(f'KP{i}_x', 0.0)
                    y = row.get(f'KP{i}_y', 0.0)
                    c = row.get(f'KP{i}_c', 0.0)
                    line_parts.append(f"{x:.4f}")
                    line_parts.append(f"{y:.4f}")
                    line_parts.append(f"{c:.4f}")
                
                f.write(','.join(line_parts) + '\n')

        print(f"성공적으로 파일을 생성했습니다: {txt_filename}")

def process_multiple_items(data_items):
    """문자열 또는 리스트의 리스트로 제공된 데이터를 처리하는 메인 함수입니다."""
    try:
        combined_data = []
        for item in data_items:
            data_list = []
            if isinstance(item, str):
                if item.strip():
                    data_list = eval(item)
            elif isinstance(item, list):
                data_list = item
            elif hasattr(item, '__iter__') and not isinstance(item, str):  # SimpleNamespace 객체들의 리스트 처리
                data_list = list(item)
            else:
                from types import SimpleNamespace
                if isinstance(item, SimpleNamespace):
                    data_list = [item]
                else:
                    print(f"처리할 수 없는 데이터 타입입니다: {type(item)}. 건너뜁니다.")
                    continue
                
            combined_data.extend(data_list)
        
        processed_data = process_results(combined_data)
        save_to_csv(processed_data)

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc() # 상세한 오류 추적을 위해 추가
        print("데이터의 형식을 확인해 주세요.")

# --- 메인 스크립트 실행 ---

# 첫 번째 데이터 문자열
data_string_1 = [{'input_center': array([532.81055, 813.72205], dtype=float32), 'id': 1, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 1, 'category_id': array(0), 'input_scale': array([586.95624, 782.60834], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 1, 'image_id': 1, 'category_id': 0, 'iscrowd': 0, 'bbox': [298.028, 755.885, 469.56499999999994, 115.67399999999998], 'area': 54316.461809999986, 'num_keypoints': 5, 'keypoints': [767.593, 871.559, 2, 746.978, 755.885, 2, 643.902, 824.602, 1, 462.948, 840.636, 1, 298.028, 871.559, 1, 346.129, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[586.95624, 782.60834]], dtype=float32), 'bboxes': array([[298.028, 755.885, 767.593, 871.559]], dtype=float32), 'keypoints': array([[[ 767.593,  871.559],
        [ 746.978,  755.885],
        [ 643.902,  824.602],
        [ 462.948,  840.636],
        [ 298.028,  871.559],
        [ 346.129, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.8967535 , 1.006735  , 0.7887745 , 0.71093154, 0.4000771 ,
        0.24920794]], dtype=float32), 'bboxes': array([[298.028, 755.885, 767.593, 871.559]], dtype=float32), 'keypoints': array([[[ 755.9762 ,  871.8063 ],
        [ 737.6338 ,  749.5237 ],
        [ 639.8078 ,  829.0074 ],
        [ 462.49808,  847.34973],
        [ 309.6449 ,  877.9204 ],
        [ 279.07425, 1091.9148 ]]], dtype=float32), 'keypoints_visible': array([[0.8967535 , 1.006735  , 0.7887745 , 0.71093154, 0.4000771 ,
        0.24920794]], dtype=float32)}}, {'input_center': array([1579.64   ,  607.04846], dtype=float32), 'id': 2, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 1, 'category_id': array(1), 'input_scale': array([495.4425, 660.59  ], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 2, 'image_id': 1, 'category_id': 1, 'iscrowd': 0, 'bbox': [1381.463, 487.89, 396.35400000000004, 238.317], 'area': 94457.89621800001, 'num_keypoints': 6, 'keypoints': [1413.531, 644.793, 1, 1381.463, 502.778, 1, 1473.086, 507.359, 1, 1573.871, 494.761, 1, 1628.844, 487.89, 1, 1777.817, 726.207, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[495.4425, 660.59  ]], dtype=float32), 'bboxes': array([[1381.463,  487.89 , 1777.817,  726.207]], dtype=float32), 'keypoints': array([[[1413.531,  644.793],
        [1381.463,  502.778],
        [1473.086,  507.359],
        [1573.871,  494.761],
        [1628.844,  487.89 ],
        [1777.817,  726.207]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.10227482, 0.24170005, 0.48536426, 0.5845014 , 0.3506528 ,
        0.2616961 ]], dtype=float32), 'bboxes': array([[1381.463,  487.89 , 1777.817,  726.207]], dtype=float32), 'keypoints': array([[[1360.3035 ,  434.15967],
        [1342.2405 ,  516.7334 ],
        [1458.3599 ,  501.25085],
        [1638.9899 ,  485.76825],
        [1737.0463 ,  511.57257],
        [1762.8505 ,  738.6504 ]]], dtype=float32), 'keypoints_visible': array([[0.10227482, 0.24170005, 0.48536426, 0.5845014 , 0.3506528 ,
        0.2616961 ]], dtype=float32)}}, {'input_center': array([617.47754, 713.72205], dtype=float32), 'id': 7, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 2, 'category_id': array(0), 'input_scale': array([661.95624, 882.60834], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 7, 'image_id': 2, 'category_id': 0, 'iscrowd': 0, 'bbox': [352.695, 606.552, 529.565, 214.34000000000003], 'area': 113506.96210000003, 'num_keypoints': 5, 'keypoints': [882.26, 801.559, 2, 750.311, 606.552, 2, 662.569, 759.269, 1, 506.948, 745.303, 1, 352.695, 820.892, 1, 454.129, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[661.95624, 882.60834]], dtype=float32), 'bboxes': array([[352.695, 606.552, 882.26 , 820.892]], dtype=float32), 'keypoints': array([[[ 882.26 ,  801.559],
        [ 750.311,  606.552],
        [ 662.569,  759.269],
        [ 506.948,  745.303],
        [ 352.695,  820.892],
        [ 454.129, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.97118545, 0.89514804, 0.6663474 , 0.7173921 , 0.71212125,
        0.41583145]], dtype=float32), 'bboxes': array([[352.695, 606.552, 882.26 , 820.892]], dtype=float32), 'keypoints': array([[[ 882.9496 ,  799.91425],
        [ 751.9374 ,  606.8437 ],
        [ 655.4021 ,  758.542  ],
        [ 496.80844,  772.33276],
        [ 372.69165,  827.4958 ],
        [ 427.85464, 1027.4617 ]]], dtype=float32), 'keypoints_visible': array([[0.97118545, 0.89514804, 0.6663474 , 0.7173921 , 0.71212125,
        0.41583145]], dtype=float32)}}, {'input_center': array([1574.3065 ,  506.71497], dtype=float32), 'id': 8, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 2, 'category_id': array(1), 'input_scale': array([542.10876, 722.8117 ], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 8, 'image_id': 2, 'category_id': 1, 'iscrowd': 0, 'bbox': [1357.463, 383.223, 433.6870000000001, 246.98399999999998], 'area': 107113.75000800002, 'num_keypoints': 6, 'keypoints': [1395.531, 567.46, 1, 1357.463, 476.778, 1, 1447.753, 443.359, 1, 1529.204, 418.094, 1, 1629.511, 383.223, 1, 1791.15, 630.207, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[542.10876, 722.8117 ]], dtype=float32), 'bboxes': array([[1357.463,  383.223, 1791.15 ,  630.207]], dtype=float32), 'keypoints': array([[[1395.531,  567.46 ],
        [1357.463,  476.778],
        [1447.753,  443.359],
        [1529.204,  418.094],
        [1629.511,  383.223],
        [1791.15 ,  630.207]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.14920686, 0.34902075, 0.6644679 , 0.40358818, 0.32149935,
        0.40312135]], dtype=float32), 'bboxes': array([[1357.463,  383.223, 1791.15 ,  630.207]], dtype=float32), 'keypoints': array([[[1314.546  ,  458.71576],
        [1373.8392 ,  441.77484],
        [1492.4255 ,  413.54   ],
        [1611.0118 ,  385.30518],
        [1678.7754 ,  390.95215],
        [1752.1859 ,  616.8308 ]]], dtype=float32), 'keypoints_visible': array([[0.14920686, 0.34902075, 0.6644679 , 0.40358818, 0.32149935,
        0.40312135]], dtype=float32)}}, {'input_center': array([620.72754, 707.22205], dtype=float32), 'id': 13, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 3, 'category_id': array(0), 'input_scale': array([630.08124, 840.10834], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 13, 'image_id': 3, 'category_id': 0, 'iscrowd': 0, 'bbox': [368.695, 606.052, 504.065, 202.34000000000003], 'area': 101992.51210000002, 'num_keypoints': 5, 'keypoints': [872.76, 782.059, 2, 746.811, 606.052, 2, 660.569, 755.269, 1, 513.448, 742.803, 1, 368.695, 808.392, 1, 463.129, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_58.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[630.08124, 840.10834]], dtype=float32), 'bboxes': array([[368.695, 606.052, 872.76 , 808.392]], dtype=float32), 'keypoints': array([[[ 872.76 ,  782.059],
        [ 746.811,  606.052],
        [ 660.569,  755.269],
        [ 513.448,  742.803],
        [ 368.695,  808.392],
        [ 463.129, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.9875807 , 0.8922993 , 0.659559  , 0.73131263, 0.65830827,
        0.42235008]], dtype=float32), 'bboxes': array([[368.695, 606.052, 872.76 , 808.392]], dtype=float32), 'keypoints': array([[[ 866.853  ,  782.7005 ],
        [ 748.71277,  605.4902 ],
        [ 656.8259 ,  756.44714],
        [ 505.869  ,  769.57385],
        [ 387.72876,  822.0806 ],
        [ 446.79886, 1018.981  ]]], dtype=float32), 'keypoints_visible': array([[0.9875807 , 0.8922993 , 0.659559  , 0.73131263, 0.65830827,
        0.42235008]], dtype=float32)}}, {'input_center': array([1559.3406 ,  501.38202], dtype=float32), 'id': 14, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 3, 'category_id': array(1), 'input_scale': array([531.1913 , 708.25507], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 14, 'image_id': 3, 'category_id': 1, 'iscrowd': 0, 'bbox': [1346.864, 375.89, 424.953, 250.98400000000004], 'area': 106656.40375200001, 'num_keypoints': 6, 'keypoints': [1346.864, 540.793, 1, 1353.463, 438.778, 1, 1447.086, 423.359, 1, 1522.537, 404.094, 1, 1607.511, 375.89, 1, 1771.817, 626.874, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_58.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[531.1913 , 708.25507]], dtype=float32), 'bboxes': array([[1346.864,  375.89 , 1771.817,  626.874]], dtype=float32), 'keypoints': array([[[1346.864,  540.793],
        [1353.463,  438.778],
        [1447.086,  423.359],
        [1522.537,  404.094],
        [1607.511,  375.89 ],
        [1771.817,  626.874]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.11824843, 0.34524435, 0.59934044, 0.4742903 , 0.37346092,
        0.43697476]], dtype=float32), 'bboxes': array([[1346.864,  375.89 , 1771.817,  626.874]], dtype=float32), 'keypoints': array([[[1329.711  ,  354.7511 ],
        [1357.3772 ,  432.2165 ],
        [1468.0421 ,  410.08353],
        [1584.2401 ,  382.4173 ],
        [1700.4382 ,  415.61676],
        [1717.038  ,  603.747  ]]], dtype=float32), 'keypoints_visible': array([[0.11824843, 0.34524435, 0.59934044, 0.4742903 , 0.37346092,
        0.43697476]], dtype=float32)}}, {'input_center': array([619.39404, 706.22205], dtype=float32), 'id': 19, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 4, 'category_id': array(0), 'input_scale': array([643.415  , 857.88666], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 19, 'image_id': 4, 'category_id': 0, 'iscrowd': 0, 'bbox': [362.028, 602.052, 514.732, 208.34000000000003], 'area': 107239.26488000002, 'num_keypoints': 5, 'keypoints': [876.76, 783.392, 2, 748.811, 602.052, 2, 653.902, 747.936, 1, 504.781, 744.803, 1, 362.028, 810.392, 1, 467.796, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_87.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[643.415  , 857.88666]], dtype=float32), 'bboxes': array([[362.028, 602.052, 876.76 , 810.392]], dtype=float32), 'keypoints': array([[[ 876.76 ,  783.392],
        [ 748.811,  602.052],
        [ 653.902,  747.936],
        [ 504.781,  744.803],
        [ 362.028,  810.392],
        [ 467.796, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.9272813 , 0.8821781 , 0.6477651 , 0.74464095, 0.7203604 ,
        0.39334685]], dtype=float32), 'bboxes': array([[362.028, 602.052, 876.76 , 810.392]], dtype=float32), 'keypoints': array([[[ 864.02576,  783.2978 ],
        [ 750.0877 ,  609.03955],
        [ 662.9586 ,  756.48883],
        [ 508.8071 ,  763.1911 ],
        [ 381.46454,  816.809  ],
        [ 448.48694, 1024.5784 ]]], dtype=float32), 'keypoints_visible': array([[0.9272813 , 0.8821781 , 0.6477651 , 0.74464095, 0.7203604 ,
        0.39334685]], dtype=float32)}}, {'input_center': array([1526.3406,  503.049 ], dtype=float32), 'id': 20, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 4, 'category_id': array(1), 'input_scale': array([560.35876, 747.145  ], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 20, 'image_id': 4, 'category_id': 1, 'iscrowd': 0, 'bbox': [1302.197, 380.557, 448.28700000000003, 244.98400000000004], 'area': 109823.14240800003, 'num_keypoints': 6, 'keypoints': [1302.197, 392.793, 1, 1314.13, 401.445, 1, 1467.086, 409.359, 1, 1529.87, 406.094, 1, 1576.844, 380.557, 1, 1750.484, 625.541, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_87.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[560.35876, 747.145  ]], dtype=float32), 'bboxes': array([[1302.197,  380.557, 1750.484,  625.541]], dtype=float32), 'keypoints': array([[[1302.197,  392.793],
        [1314.13 ,  401.445],
        [1467.086,  409.359],
        [1529.87 ,  406.094],
        [1576.844,  380.557],
        [1750.484,  625.541]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.41205648, 0.5925001 , 0.7216774 , 0.6504332 , 0.3692509 ,
        0.41242403]], dtype=float32), 'bboxes': array([[1302.197,  380.557, 1750.484,  625.541]], dtype=float32), 'keypoints': array([[[1278.2651 ,  377.552  ],
        [1348.3099 ,  418.4115 ],
        [1453.3772 ,  400.90027],
        [1581.7927 ,  371.71494],
        [1681.023  ,  395.0632 ],
        [1716.0454 ,  599.36066]]], dtype=float32), 'keypoints_visible': array([[0.41205648, 0.5925001 , 0.7216774 , 0.6504332 , 0.3692509 ,
        0.41242403]], dtype=float32)}}, {'input_center': array([626.39404, 704.97205], dtype=float32), 'id': 25, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 5, 'category_id': array(0), 'input_scale': array([642.165, 856.22 ], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 25, 'image_id': 5, 'category_id': 0, 'iscrowd': 0, 'bbox': [369.528, 595.052, 513.732, 219.84000000000003], 'area': 112938.84288000001, 'num_keypoints': 5, 'keypoints': [883.26, 778.392, 2, 749.811, 595.052, 2, 664.902, 750.936, 1, 514.781, 746.803, 1, 369.528, 814.892, 1, 480.296, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_116.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[642.165, 856.22 ]], dtype=float32), 'bboxes': array([[369.528, 595.052, 883.26 , 814.892]], dtype=float32), 'keypoints': array([[[ 883.26 ,  778.392],
        [ 749.811,  595.052],
        [ 664.902,  750.936],
        [ 514.781,  746.803],
        [ 369.528,  814.892],
        [ 480.296, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.9573953 , 0.88644457, 0.6296545 , 0.7591512 , 0.76031625,
        0.36664632]], dtype=float32), 'bboxes': array([[369.528, 595.052, 883.26 , 814.892]], dtype=float32), 'keypoints': array([[[ 877.23975,  781.8981 ],
        [ 756.8338 ,  601.2892 ],
        [ 669.87396,  755.1412 ],
        [ 529.4004 ,  768.5196 ],
        [ 395.616  ,  815.3442 ],
        [ 475.88663, 1016.0207 ]]], dtype=float32), 'keypoints_visible': array([[0.9573953 , 0.88644457, 0.6296545 , 0.7591512 , 0.76031625,
        0.36664632]], dtype=float32)}}, {'input_center': array([1514.64   ,  298.34198], dtype=float32), 'id': 26, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 5, 'category_id': array(1), 'input_scale': array([599.61, 799.48], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 26, 'image_id': 5, 'category_id': 1, 'iscrowd': 0, 'bbox': [1274.796, 177.224, 479.6879999999999, 242.236], 'area': 116197.70236799997, 'num_keypoints': 6, 'keypoints': [1304.864, 419.46, 1, 1274.796, 364.111, 1, 1341.753, 284.692, 1, 1451.203, 247.427, 1, 1530.844, 177.224, 1, 1754.484, 346.208, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_116.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[599.61, 799.48]], dtype=float32), 'bboxes': array([[1274.796,  177.224, 1754.484,  419.46 ]], dtype=float32), 'keypoints': array([[[1304.864,  419.46 ],
        [1274.796,  364.111],
        [1341.753,  284.692],
        [1451.203,  247.427],
        [1530.844,  177.224],
        [1754.484,  346.208]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.32359067, 0.53641456, 0.5248139 , 0.42969602, 0.6597142 ,
        0.4080745 ]], dtype=float32), 'bboxes': array([[1274.796,  177.224, 1754.484,  419.46 ]], dtype=float32), 'keypoints': array([[[1280.4174 ,  420.13776],
        [1299.1552 ,  338.94058],
        [1380.3524 ,  251.49745],
        [1492.7792 ,  189.03807],
        [1580.2224 ,  145.31651],
        [1711.3871 ,  307.71088]]], dtype=float32), 'keypoints_visible': array([[0.32359067, 0.53641456, 0.5248139 , 0.42969602, 0.6597142 ,
        0.4080745 ]], dtype=float32)}}, {'input_center': array([636.144  , 704.88855], dtype=float32), 'id': 31, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 6, 'category_id': array(0), 'input_scale': array([615.7075 , 820.94336], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 31, 'image_id': 6, 'category_id': 0, 'iscrowd': 0, 'bbox': [389.861, 594.385, 492.56600000000003, 221.00700000000006], 'area': 108860.53396200003, 'num_keypoints': 5, 'keypoints': [882.427, 772.892, 2, 755.644, 594.385, 2, 690.069, 764.603, 1, 544.614, 750.803, 1, 389.861, 815.392, 1, 494.963, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_145.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[615.7075 , 820.94336]], dtype=float32), 'bboxes': array([[389.861, 594.385, 882.427, 815.392]], dtype=float32), 'keypoints': array([[[ 882.427,  772.892],
        [ 755.644,  594.385],
        [ 690.069,  764.603],
        [ 544.614,  750.803],
        [ 389.861,  815.392],
        [ 494.963, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.96473   , 0.8793707 , 0.6610416 , 0.7448591 , 0.64838386,
        0.36786628]], dtype=float32), 'bboxes': array([[389.861, 594.385, 882.427, 815.392]], dtype=float32), 'keypoints': array([[[ 876.6547 ,  772.23157],
        [ 754.79596,  599.06384],
        [ 671.4189 ,  765.81793],
        [ 536.73285,  765.81793],
        [ 408.46048,  823.5405 ],
        [ 472.59668, 1022.36273]]], dtype=float32), 'keypoints_visible': array([[0.96473   , 0.8793707 , 0.6610416 , 0.7448591 , 0.64838386,
        0.36786628]], dtype=float32)}}, {'input_center': array([1516.6741,  337.676 ], dtype=float32), 'id': 32, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 6, 'category_id': array(1), 'input_scale': array([561.19006, 748.2534 ], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 32, 'image_id': 6, 'category_id': 1, 'iscrowd': 0, 'bbox': [1292.198, 211.558, 448.952, 252.236], 'area': 113241.856672, 'num_keypoints': 6, 'keypoints': [1292.198, 463.794, 1, 1304.796, 429.445, 1, 1365.086, 361.025, 1, 1448.87, 310.094, 1, 1520.178, 211.558, 1, 1741.15, 334.875, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_145.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[561.19006, 748.2534 ]], dtype=float32), 'bboxes': array([[1292.198,  211.558, 1741.15 ,  463.794]], dtype=float32), 'keypoints': array([[[1292.198,  463.794],
        [1304.796,  429.445],
        [1365.086,  361.025],
        [1448.87 ,  310.094],
        [1520.178,  211.558],
        [1741.15 ,  334.875]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.22890644, 0.22805935, 0.33647722, 0.42736378, 0.42089736,
        0.35350585]], dtype=float32), 'bboxes': array([[1292.198,  211.558, 1741.15 ,  463.794]], dtype=float32), 'keypoints': array([[[1247.7705 ,  422.4391 ],
        [1326.6879 ,  387.3647 ],
        [1350.0708 ,  276.29584],
        [1449.4481 ,  211.9928 ],
        [1531.2885 ,  159.38124],
        [1706.6603 ,  211.9928 ]]], dtype=float32), 'keypoints_visible': array([[0.22890644, 0.22805935, 0.33647722, 0.42736378, 0.42089736,
        0.35350585]], dtype=float32)}}, {'input_center': array([638.144  , 706.88855], dtype=float32), 'id': 37, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 7, 'category_id': array(0), 'input_scale': array([610.7075, 814.2767], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 37, 'image_id': 7, 'category_id': 0, 'iscrowd': 0, 'bbox': [393.861, 598.385, 488.56600000000003, 217.00700000000006], 'area': 106022.24196200004, 'num_keypoints': 5, 'keypoints': [882.427, 769.892, 2, 753.644, 598.385, 2, 679.069, 762.603, 1, 553.614, 750.803, 1, 393.861, 815.392, 1, 487.963, 1080.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_174.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[610.7075, 814.2767]], dtype=float32), 'bboxes': array([[393.861, 598.385, 882.427, 815.392]], dtype=float32), 'keypoints': array([[[ 882.427,  769.892],
        [ 753.644,  598.385],
        [ 679.069,  762.603],
        [ 553.614,  750.803],
        [ 393.861,  815.392],
        [ 487.963, 1080.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.95983547, 0.88117456, 0.64922655, 0.7462854 , 0.6140966 ,
        0.36423603]], dtype=float32), 'bboxes': array([[393.861, 598.385, 882.427, 815.392]], dtype=float32), 'keypoints': array([[[ 876.7016 ,  767.3231 ],
        [ 755.8324 ,  601.9232 ],
        [ 673.13245,  767.3231 ],
        [ 539.54016,  767.3231 ],
        [ 412.30942,  830.93854],
        [ 488.64786, 1021.7846 ]]], dtype=float32), 'keypoints_visible': array([[0.95983547, 0.88117456, 0.64922655, 0.7462854 , 0.6140966 ,
        0.36423603]], dtype=float32)}}, {'input_center': array([1489.6741,  397.009 ], dtype=float32), 'id': 38, 'dataset_name': 'hands_only_dataset', 'pad_shape': (256, 192), 'img_id': 7, 'category_id': array(1), 'input_scale': array([595.35754, 793.81006], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5], 'raw_ann_info': {'id': 38, 'image_id': 7, 'category_id': 1, 'iscrowd': 0, 'bbox': [1251.531, 245.891, 476.28600000000006, 302.236], 'area': 143950.77549600002, 'num_keypoints': 6, 'keypoints': [1251.531, 548.127, 2, 1334.796, 494.778, 1, 1388.42, 437.359, 1, 1446.536, 372.76, 1, 1509.511, 245.891, 1, 1727.817, 323.541, 2]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_174.png', 'ori_shape': (1080, 1920), 'input_size': (192, 256), 'batch_input_shape': (256, 192), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[595.35754, 793.81006]], dtype=float32), 'bboxes': array([[1251.531,  245.891, 1727.817,  548.127]], dtype=float32), 'keypoints': array([[[1251.531,  548.127],
        [1334.796,  494.778],
        [1388.42 ,  437.359],
        [1446.536,  372.76 ],
        [1509.511,  245.891],
        [1727.817,  323.541]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 1., 1., 1.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.8571904 , 0.7696545 , 0.35585093, 0.5996415 , 0.6176741 ,
        0.55292696]], dtype=float32), 'bboxes': array([[1251.531,  245.891, 1727.817,  548.127]], dtype=float32), 'keypoints': array([[[1250.9109 ,  555.1509 ],
        [1331.5322 ,  486.9328 ],
        [1387.347  ,  350.4967 ],
        [1449.3634 ,  276.077  ],
        [1498.9766 ,  232.66551],
        [1685.0258 ,  214.0606 ]]], dtype=float32), 'keypoints_visible': array([[0.8571904 , 0.7696545 , 0.35585093, 0.5996415 , 0.6176741 ,
        0.55292696]], dtype=float32)}}]

# 두 번째 데이터 문자열
data_string_2 = [{'input_center': array([411.9835, 121.0405], dtype=float32), 'id': 3, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 1, 'category_id': array(3), 'input_scale': array([105.83374, 105.83374], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 3, 'image_id': 1, 'category_id': 3, 'iscrowd': 0, 'bbox': [369.65, 119.874, 84.66700000000003, 2.3329999999999984], 'area': 197.52811099999994, 'num_keypoints': 2, 'keypoints': [369.65, 122.207, 2, 454.317, 119.874, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[105.83374, 105.83374]], dtype=float32), 'bboxes': array([[369.65 , 119.874, 454.317, 122.207]], dtype=float32), 'keypoints': array([[[369.65 , 122.207],
        [454.317, 119.874],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.71447533, 0.44782057, 0.02080851, 0.02970348, 0.07642575,
        0.00673619, 0.01299551, 0.05013225, 0.03782471, 0.00549155,
        0.02262846]], dtype=float32), 'bboxes': array([[369.65 , 119.874, 454.317, 122.207]], dtype=float32), 'keypoints': array([[[367.7483  , 123.93439 ],
        [454.56503 , 120.62708 ],
        [411.57007 , 161.96838 ],
        [454.56503 , 120.62708 ],
        [367.7483  , 123.10756 ],
        [452.91138 , 127.24169 ],
        [363.61417 , 172.30371 ],
        [367.7483  , 123.10756 ],
        [454.56503 , 120.62708 ],
        [451.25772 , 128.06851 ],
        [367.7483  , 124.761215]]], dtype=float32), 'keypoints_visible': array([[0.71447533, 0.44782057, 0.02080851, 0.02970348, 0.07642575,
        0.00673619, 0.01299551, 0.05013225, 0.03782471, 0.00549155,
        0.02262846]], dtype=float32)}}, {'input_center': array([828.3165, 490.124 ], dtype=float32), 'id': 4, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 1, 'category_id': array(4), 'input_scale': array([98.54118, 98.54118], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 4, 'image_id': 1, 'category_id': 4, 'iscrowd': 0, 'bbox': [788.9, 480.541, 78.83299999999997, 19.165999999999997], 'area': 1510.913277999999, 'num_keypoints': 3, 'keypoints': [789.066, 486.54, 2, 788.9, 480.541, 2, 867.733, 499.707, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[98.54118, 98.54118]], dtype=float32), 'bboxes': array([[788.9  , 480.541, 867.733, 499.707]], dtype=float32), 'keypoints': array([[[789.066, 486.54 ],
        [788.9  , 480.541],
        [867.733, 499.707],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.50347483, 0.62605613, 0.5749595 , 0.04632011, 0.08727345,
        0.0085625 , 0.00829967, 0.03755841, 0.05510679, 0.00514625,
        0.01032071]], dtype=float32), 'bboxes': array([[788.9  , 480.541, 867.733, 499.707]], dtype=float32), 'keypoints': array([[[789.43896, 480.50082],
        [789.43896, 483.58023],
        [861.8051 , 500.517  ],
        [789.43896, 484.3501 ],
        [789.43896, 482.04053],
        [795.5978 , 481.2707 ],
        [837.9397 , 492.81848],
        [790.2088 , 478.96112],
        [789.43896, 482.8104 ],
        [783.28015, 478.19128],
        [788.6691 , 479.731  ]]], dtype=float32), 'keypoints_visible': array([[0.50347483, 0.62605613, 0.5749595 , 0.04632011, 0.08727345,
        0.0085625 , 0.00829967, 0.03755841, 0.05510679, 0.00514625,
        0.01032071]], dtype=float32)}}, {'input_center': array([797.15 , 484.582], dtype=float32), 'id': 5, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 1, 'category_id': array(5), 'input_scale': array([28.4375, 28.4375], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 5, 'image_id': 1, 'category_id': 5, 'iscrowd': 0, 'bbox': [792.317, 473.207, 9.66599999999994, 22.75], 'area': 219.90149999999863, 'num_keypoints': 3, 'keypoints': [792.317, 491.124, 2, 797.733, 495.957, 2, 801.983, 473.207, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[28.4375, 28.4375]], dtype=float32), 'bboxes': array([[792.317, 473.207, 801.983, 495.957]], dtype=float32), 'keypoints': array([[[792.317, 491.124],
        [797.733, 495.957],
        [801.983, 473.207],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.4061264 , 0.06472491, 0.5881753 , 0.01321483, 0.04078862,
        0.0054086 , 0.00440693, 0.03711341, 0.00486188, 0.00544379,
        0.00362193]], dtype=float32), 'bboxes': array([[792.317, 473.207, 801.983, 495.957]], dtype=float32), 'keypoints': array([[[797.70544, 494.2463 ],
        [797.70544, 494.9128 ],
        [800.37146, 473.5847 ],
        [800.37146, 473.5847 ],
        [797.70544, 494.2463 ],
        [797.70544, 492.9133 ],
        [801.4823 , 471.58517],
        [797.4833 , 494.2463 ],
        [797.70544, 495.13498],
        [798.5941 , 472.25168],
        [792.81775, 488.0256 ]]], dtype=float32), 'keypoints_visible': array([[0.4061264 , 0.06472491, 0.5881753 , 0.01321483, 0.04078862,
        0.0054086 , 0.00440693, 0.03711341, 0.00486188, 0.00544379,
        0.00362193]], dtype=float32)}}, {'input_center': array([1315.3165 ,  165.29099], dtype=float32), 'id': 6, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 1, 'category_id': array(2), 'input_scale': array([299.99997, 299.99997], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 6, 'image_id': 1, 'category_id': 2, 'iscrowd': 0, 'bbox': [1271.65, 45.291, 87.33299999999986, 240.0], 'area': 20959.919999999966, 'num_keypoints': 3, 'keypoints': [1279.983, 45.291, 2, 1271.65, 52.624, 2, 1358.983, 285.291, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_0.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[299.99997, 299.99997]], dtype=float32), 'bboxes': array([[1271.65 ,   45.291, 1358.983,  285.291]], dtype=float32), 'keypoints': array([[[1279.983,   45.291],
        [1271.65 ,   52.624],
        [1358.983,  285.291],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.96534395, 0.9098752 , 0.96821284, 0.05522075, 0.13277212,
        0.0081977 , 0.00505652, 0.03581136, 0.09419994, 0.0084129 ,
        0.03618936]], dtype=float32), 'bboxes': array([[1271.65 ,   45.291, 1358.983,  285.291]], dtype=float32), 'keypoints': array([[[1278.9884 ,   46.93162],
        [1271.9572 ,   51.61912],
        [1361.0197 ,  285.9941 ],
        [1264.9259 ,   53.96287],
        [1274.3009 ,   51.61912],
        [1290.7072 ,   53.96287],
        [1353.9884 ,  310.6035 ],
        [1281.3322 ,   39.90037],
        [1271.9572 ,   51.61912],
        [1382.1134 ,  285.9941 ],
        [1276.6447 ,   46.93162]]], dtype=float32), 'keypoints_visible': array([[0.96534395, 0.9098752 , 0.96821284, 0.05522075, 0.13277212,
        0.0081977 , 0.00505652, 0.03581136, 0.09419994, 0.0084129 ,
        0.03618936]], dtype=float32)}}, {'input_center': array([411.90802, 121.1915 ], dtype=float32), 'id': 9, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 2, 'category_id': array(3), 'input_scale': array([105.96249, 105.96249], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 9, 'image_id': 2, 'category_id': 3, 'iscrowd': 0, 'bbox': [369.523, 119.993, 84.76999999999998, 2.3970000000000056], 'area': 203.19369000000043, 'num_keypoints': 2, 'keypoints': [369.523, 122.39, 2, 454.293, 119.993, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[105.96249, 105.96249]], dtype=float32), 'bboxes': array([[369.523, 119.993, 454.293, 122.39 ]], dtype=float32), 'keypoints': array([[[369.523, 122.39 ],
        [454.293, 119.993],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.7012886 , 0.46496564, 0.02457096, 0.03160648, 0.07544041,
        0.00655355, 0.01281568, 0.05030705, 0.03888974, 0.00550482,
        0.01968814]], dtype=float32), 'bboxes': array([[369.523, 119.993, 454.293, 122.39 ]], dtype=float32), 'keypoints': array([[[368.44684 , 124.08891 ],
        [454.54135 , 120.77758 ],
        [401.56012 , 162.16917 ],
        [453.71353 , 121.605415],
        [367.61902 , 123.26108 ],
        [450.4022  , 125.744576],
        [364.30768 , 172.51707 ],
        [368.44684 , 123.26108 ],
        [454.54135 , 120.77758 ],
        [447.9187  , 124.08891 ],
        [367.61902 , 124.08891 ]]], dtype=float32), 'keypoints_visible': array([[0.7012886 , 0.46496564, 0.02457096, 0.03160648, 0.07544041,
        0.00655355, 0.01281568, 0.05030705, 0.03888974, 0.00550482,
        0.01968814]], dtype=float32)}}, {'input_center': array([840.004 , 482.0405], dtype=float32), 'id': 10, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 2, 'category_id': array(4), 'input_scale': array([93.69743, 93.69743], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 10, 'image_id': 2, 'category_id': 4, 'iscrowd': 0, 'bbox': [802.525, 475.624, 74.95799999999997, 12.83299999999997], 'area': 961.9360139999974, 'num_keypoints': 3, 'keypoints': [803.608, 482.54, 2, 802.525, 475.624, 2, 877.483, 488.457, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[93.69743, 93.69743]], dtype=float32), 'bboxes': array([[802.525, 475.624, 877.483, 488.457]], dtype=float32), 'keypoints': array([[[803.608, 482.54 ],
        [802.525, 475.624],
        [877.483, 488.457],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.59181184, 0.76557   , 0.47560072, 0.06337708, 0.08811282,
        0.00854253, 0.00707605, 0.04061999, 0.07390614, 0.00441715,
        0.01096086]], dtype=float32), 'bboxes': array([[802.525, 475.624, 877.483, 488.457]], dtype=float32), 'keypoints': array([[[804.50146, 480.94247],
        [803.7695 , 475.0864 ],
        [879.1666 , 486.06656],
        [803.7695 , 475.0864 ],
        [803.7695 , 475.0864 ],
        [803.7695 , 486.79858],
        [801.5734 , 497.04672],
        [804.50146, 481.6745 ],
        [803.7695 , 475.0864 ],
        [812.5536 , 496.31473],
        [803.7695 , 480.94247]]], dtype=float32), 'keypoints_visible': array([[0.59181184, 0.76557   , 0.47560072, 0.06337708, 0.08811282,
        0.00854253, 0.00707605, 0.04061999, 0.07390614, 0.00441715,
        0.01096086]], dtype=float32)}}, {'input_center': array([812.6085, 483.2695], dtype=float32), 'id': 11, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 2, 'category_id': array(5), 'input_scale': array([39.21875, 39.21875], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 11, 'image_id': 2, 'category_id': 5, 'iscrowd': 0, 'bbox': [810.234, 467.582, 4.74899999999991, 31.375], 'area': 148.9998749999972, 'num_keypoints': 3, 'keypoints': [810.234, 481.124, 2, 812.483, 498.957, 2, 814.983, 467.582, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[39.21875, 39.21875]], dtype=float32), 'bboxes': array([[810.234, 467.582, 814.983, 498.957]], dtype=float32), 'keypoints': array([[[810.234, 481.124],
        [812.483, 498.957],
        [814.983, 467.582],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.33123308, 0.2215944 , 0.24271572, 0.02269963, 0.02754207,
        0.00460583, 0.00447684, 0.01843587, 0.01408522, 0.00496102,
        0.00680611]], dtype=float32), 'bboxes': array([[810.234, 467.582, 814.983, 498.957]], dtype=float32), 'keypoints': array([[[811.53613, 498.12973],
        [815.5193 , 469.63486],
        [813.6809 , 469.63486],
        [815.8257 , 469.63486],
        [811.22974, 498.43613],
        [812.7617 , 498.12973],
        [801.11865, 502.26608],
        [811.53613, 498.12973],
        [815.5193 , 469.63486],
        [805.1018 , 483.1163 ],
        [811.53613, 498.12973]]], dtype=float32), 'keypoints_visible': array([[0.33123308, 0.2215944 , 0.24271572, 0.02269963, 0.02754207,
        0.00460583, 0.00447684, 0.01843587, 0.01408522, 0.00496102,
        0.00680611]], dtype=float32)}}, {'input_center': array([1315.492,  165.161], dtype=float32), 'id': 12, 'dataset_name': 'surgical_tools', 'pad_shape': (256, 256), 'img_id': 2, 'category_id': array(2), 'input_scale': array([299.77002, 299.77002], dtype=float32), 'flip_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ori_shape': (1080, 1920), 'raw_ann_info': {'id': 12, 'image_id': 2, 'category_id': 2, 'iscrowd': 0, 'bbox': [1271.779, 45.253, 87.42599999999993, 239.81600000000003], 'area': 20966.153615999985, 'num_keypoints': 3, 'keypoints': [1279.96, 45.253, 2, 1271.779, 52.617, 2, 1359.205, 285.069, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0]}, 'img_path': 'data/hand_tool_dataset/val/E66F_frame_29.png', 'input_size': (256, 256), 'batch_input_shape': (256, 256), 'img_shape': (1080, 1920), 'gt_instance_labels': {}, 'gt_instances': {'bbox_scores': array([1.], dtype=float32), 'bbox_scales': array([[299.77002, 299.77002]], dtype=float32), 'bboxes': array([[1271.779,   45.253, 1359.205,  285.069]], dtype=float32), 'keypoints': array([[[1279.96 ,   45.253],
        [1271.779,   52.617],
        [1359.205,  285.069],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ],
        [   0.   ,    0.   ]]], dtype=float32), 'keypoints_visible': array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}, 'pred_instances': {'bbox_scores': array([1.], dtype=float32), 'keypoint_scores': array([[0.9448085 , 0.9196532 , 0.9639807 , 0.05579916, 0.1330916 ,
        0.00785557, 0.0057387 , 0.03568468, 0.09537199, 0.00843488,
        0.03555129]], dtype=float32), 'bboxes': array([[1271.779,   45.253, 1359.205,  285.069]], dtype=float32), 'keypoints': array([[[1279.1917  ,   46.892357],
        [1272.1658  ,   51.576263],
        [1361.16    ,  285.77158 ],
        [1267.4819  ,   53.918213],
        [1274.5078  ,   51.576263],
        [1290.9015  ,   53.918213],
        [1351.7922  ,  310.3621  ],
        [1281.5336  ,   39.866493],
        [1272.1658  ,   51.576263],
        [1382.2377  ,  285.77158 ],
        [1276.8497  ,   46.892357]]], dtype=float32), 'keypoints_visible': array([[0.9448085 , 0.9196532 , 0.9639807 , 0.05579916, 0.1330916 ,
        0.00785557, 0.0057387 , 0.03568468, 0.09537199, 0.00843488,
        0.03555129]], dtype=float32)}}]

# 처리할 데이터 문자열들을 리스트로 묶기
data_strings = [data_string_2, data_string_1]

# 메인 처리 함수 호출
# process_multiple_items(data_strings)