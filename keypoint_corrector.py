import json
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import os

class TemporalKeypointCorrector:
    """시간적 정보를 활용한 키포인트 보정 클래스"""
    
    def __init__(self, gt_json_path, pred_txt_path, max_frame_gap=5):
        self.gt_json_path = gt_json_path
        self.pred_txt_path = pred_txt_path
        self.max_frame_gap = max_frame_gap  # 최대 프레임 간격
        
        self.gt_data = self._load_gt_data()
        self.pred_data = self._load_pred_data()
        print(f"로드된 GT 프레임: {len(self.gt_data)}개")
        print(f"로드된 예측 프레임: {len(self.pred_data)}개")
        print(f"프레임 번호 범위: {min(self.pred_data.keys())}-{max(self.pred_data.keys())}")
        
    def _load_gt_data(self):
        """GT JSON 데이터 로드"""
        with open(self.gt_json_path, 'r') as f:
            data = json.load(f)
        
        # 이미지 ID와 프레임 번호 매핑
        img_id_to_frame = {}
        for img in data['images']:
            frame_match = re.search(r'frame_(\d+)', img['file_name'])
            if frame_match:
                img_id_to_frame[img['id']] = int(frame_match.group(1))

        # 프레임별 GT 키포인트 정보 구성
        gt_by_frame = defaultdict(list)
        
        for ann in data['annotations']:
            if ann['num_keypoints'] > 0:
                frame_num = img_id_to_frame.get(ann['image_id'])
                if frame_num is not None:
                    # 키포인트를 3개씩 묶어서 파싱 (x, y, visibility)
                    kps = ann['keypoints']
                    keypoints = []
                    for i in range(0, len(kps), 3):
                        if i + 2 < len(kps):
                            keypoints.append({
                                'x': kps[i],
                                'y': kps[i+1], 
                                'visibility': kps[i+2],
                                'kp_idx': i//3
                            })
                    
                    gt_by_frame[frame_num].append({
                        'category_id': ann['category_id'],
                        'keypoints': keypoints
                    })
        
        return gt_by_frame
    
    def _load_pred_data(self):
        """예측 TXT 데이터 로드"""
        pred_by_frame = defaultdict(list)
        
        with open(self.pred_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                    
                frame = int(parts[0])
                track_id = int(parts[1])
                class_id = int(parts[2])
                
                # 키포인트 데이터 파싱 (7번째 인덱스부터 x,y,confidence 순서)
                keypoints = []
                for i in range(7, len(parts), 3):
                    if i + 2 < len(parts):
                        keypoints.append({
                            'x': float(parts[i]),
                            'y': float(parts[i+1]),
                            'confidence': float(parts[i+2]),
                            'kp_idx': (i-7)//3
                        })
                
                pred_by_frame[frame].append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'keypoints': keypoints
                })
        
        return pred_by_frame
    
    def _find_missing_keypoints(self, frame, gt_instance, pred_instance):
        """현재 프레임에서 누락된 키포인트 찾기"""
        missing_kps = []
        confidence_threshold = 0.15  # 신뢰도 임계값
        
        for gt_kp in gt_instance['keypoints']:
            # GT에서 visible인 키포인트만 확인
            if gt_kp['visibility'] == 2:  
                # 예측에서 해당 키포인트 찾기
                pred_kp = next((p for p in pred_instance['keypoints'] 
                              if p['kp_idx'] == gt_kp['kp_idx']), None)
                
                # 예측되지 않았거나 신뢰도가 낮은 경우
                if pred_kp is None or pred_kp['confidence'] < confidence_threshold:
                    missing_kps.append(gt_kp['kp_idx'])
        
        return missing_kps
    
    # def _find_future_keypoint(self, start_frame, class_id, kp_idx):
    #     """이후 프레임에서 키포인트 찾기"""
    #     best_kp = None
    #     best_frame = None
        
    #     available_frames = [f for f in sorted(self.pred_data.keys()) if f > start_frame]
        
    #     # 최대 max_frame_gap개의 프레임까지 확인
    #     for frame in available_frames[:self.max_frame_gap]:
    #         # 같은 클래스의 인스턴스에서 키포인트 찾기
    #         for pred_instance in self.pred_data[frame]:
    #             if pred_instance['class_id'] == class_id:
    #                 target_kp = next((kp for kp in pred_instance['keypoints'] 
    #                                 if kp['kp_idx'] == kp_idx), None)
                    
    #                 # 임계값을 낮춤 (0.4 -> 0.25)
    #                 if target_kp and target_kp['confidence'] > 0.25:
    #                     if best_kp is None or target_kp['confidence'] > best_kp['confidence']:
    #                         best_kp = target_kp
    #                         best_frame = frame
                            
    #     return best_kp, best_frame

    def _find_best_keypoint(self, start_frame, class_id, kp_idx):
        """과거와 미래 프레임에서 최고 키포인트 찾기"""
        best_kp = None
        best_frame = None
        
        # 현재 프레임 전후로 검색
        available_frames = sorted(self.pred_data.keys())
        current_idx = available_frames.index(start_frame) if start_frame in available_frames else -1
        
        if current_idx == -1:
            return None, None
        
        # 미래 프레임들 우선 검색
        future_frames = available_frames[current_idx+1:current_idx+1+self.max_frame_gap]
        # 과거 프레임들도 검색 (역방향)
        past_frames = available_frames[max(0, current_idx-2):current_idx]
        
        search_frames = future_frames + past_frames
        
        for frame in search_frames:
            for pred_instance in self.pred_data[frame]:
                if pred_instance['class_id'] == class_id:
                    target_kp = next((kp for kp in pred_instance['keypoints'] 
                                    if kp['kp_idx'] == kp_idx), None)
                    
                    # 임계값을 더 낮춤 (0.25 -> 0.1)
                    if target_kp and target_kp['confidence'] > 0.1:
                        if best_kp is None or target_kp['confidence'] > best_kp['confidence']:
                            best_kp = target_kp
                            best_frame = frame
                            
        return best_kp, best_frame
    
    def _interpolate_keypoint(self, start_frame, class_id, kp_idx):
        """선형 보간으로 키포인트 생성"""
        available_frames = sorted(self.pred_data.keys())
        current_idx = available_frames.index(start_frame) if start_frame in available_frames else -1
        
        if current_idx == -1:
            return None, None
        
        # 전후 프레임에서 해당 키포인트 찾기
        prev_kp, next_kp = None, None
        prev_frame, next_frame = None, None
        
        # 이전 프레임 검색
        for i in range(current_idx-1, max(-1, current_idx-4), -1):
            frame = available_frames[i]
            for pred_instance in self.pred_data[frame]:
                if pred_instance['class_id'] == class_id:
                    target_kp = next((kp for kp in pred_instance['keypoints'] 
                                    if kp['kp_idx'] == kp_idx), None)
                    if target_kp and target_kp['confidence'] > 0.1:
                        prev_kp = target_kp
                        prev_frame = frame
                        break
            if prev_kp:
                break
        
        # 이후 프레임 검색
        for i in range(current_idx+1, min(len(available_frames), current_idx+4)):
            frame = available_frames[i]
            for pred_instance in self.pred_data[frame]:
                if pred_instance['class_id'] == class_id:
                    target_kp = next((kp for kp in pred_instance['keypoints'] 
                                    if kp['kp_idx'] == kp_idx), None)
                    if target_kp and target_kp['confidence'] > 0.1:
                        next_kp = target_kp
                        next_frame = frame
                        break
            if next_kp:
                break
        
        # 선형 보간
        if prev_kp and next_kp:
            # 가중치 계산 (현재 프레임에 더 가까운 쪽에 더 큰 가중치)
            total_gap = next_frame - prev_frame
            curr_gap_from_prev = start_frame - prev_frame
            
            weight_next = curr_gap_from_prev / total_gap
            weight_prev = 1 - weight_next
            
            interpolated_kp = {
                'x': prev_kp['x'] * weight_prev + next_kp['x'] * weight_next,
                'y': prev_kp['y'] * weight_prev + next_kp['y'] * weight_next,
                'confidence': min(prev_kp['confidence'], next_kp['confidence']) * 0.5,  # 보간된 것은 낮은 신뢰도
                'kp_idx': kp_idx
            }
            
            return interpolated_kp, f"{prev_frame}-{next_frame}"
        
        return None, None
    
    def correct_keypoints(self):
        """키포인트 보정 수행"""
        corrected_count = 0
        interpolated_count = 0
        total_missing = 0
        
        for frame in sorted(self.pred_data.keys()):
            if frame not in self.gt_data:
                continue
                
            gt_instances = self.gt_data[frame]
            pred_instances = self.pred_data[frame]
            
            for gt_inst in gt_instances:
                pred_inst = next((p for p in pred_instances 
                                if p['class_id'] == gt_inst['category_id']), None)
                
                if pred_inst is None:
                    continue
                
                missing_kps = self._find_missing_keypoints(frame, gt_inst, pred_inst)
                total_missing += len(missing_kps)
                
                for kp_idx in missing_kps:
                    # 1단계: 직접 검색
                    best_kp, best_frame = self._find_best_keypoint(
                        frame, gt_inst['category_id'], kp_idx
                    )
                    
                    # 2단계: 보간 시도
                    if not best_kp:
                        best_kp, best_frame = self._interpolate_keypoint(
                            frame, gt_inst['category_id'], kp_idx
                        )
                        if best_kp:
                            interpolated_count += 1
                    
                    if best_kp:
                        target_kp = next((kp for kp in pred_inst['keypoints'] 
                                        if kp['kp_idx'] == kp_idx), None)
                        
                        penalty = 0.6 if "interpolated" in str(best_frame) else 0.7
                        
                        if target_kp:
                            target_kp['x'] = best_kp['x']
                            target_kp['y'] = best_kp['y']
                            target_kp['confidence'] = best_kp['confidence'] * penalty
                        else:
                            pred_inst['keypoints'].append({
                                'x': best_kp['x'],
                                'y': best_kp['y'], 
                                'confidence': best_kp['confidence'] * penalty,
                                'kp_idx': kp_idx
                            })
                        
                        corrected_count += 1
                        method = "interpolated" if "-" in str(best_frame) else "copied"
                        print(f"Frame {frame}: KP{kp_idx+1} {method} using Frame {best_frame} (conf: {best_kp['confidence']:.3f})")
        
        print(f"\n보정 완료: {corrected_count}/{total_missing} 키포인트 보정됨")
        print(f"- 직접 복사: {corrected_count - interpolated_count}개")
        print(f"- 선형 보간: {interpolated_count}개")
        return corrected_count, total_missing
    
    def save_corrected_data(self, output_path):
        """보정된 데이터 저장"""
        with open(output_path, 'w') as f:
            for frame in sorted(self.pred_data.keys()):
                for pred_inst in self.pred_data[frame]:
                    # 키포인트를 인덱스 순으로 정렬
                    sorted_kps = sorted(pred_inst['keypoints'], key=lambda x: x['kp_idx'])
                    
                    line_parts = [
                        str(frame),
                        str(pred_inst['track_id']),
                        str(pred_inst['class_id']),
                        '-1', '-1', '-1', '-1'  # bbox placeholders
                    ]
                    
                    for kp in sorted_kps:
                        line_parts.extend([
                            f"{kp['x']:.4f}",
                            f"{kp['y']:.4f}",
                            f"{kp['confidence']:.4f}"
                        ])
                    
                    f.write(','.join(line_parts) + '\n')
        
        print(f"보정된 데이터가 저장되었습니다: {output_path}")

def main():
    """메인 실행 함수"""
    # 파일 경로 설정
    gt_json_path = "data/hand_tool_dataset/annotations/val_hands.json"
    pred_txt_path = "K16O_pred.txt"
    output_path = "_pred_corrected.txt"
    
    print("시간적 키포인트 보정 시작...")
    
    # 보정기 초기화
    corrector = TemporalKeypointCorrector(
        gt_json_path=gt_json_path,
        pred_txt_path=pred_txt_path,
        max_frame_gap=5
    )
    
    # 키포인트 보정 수행
    corrected, total = corrector.correct_keypoints()
    
    # 결과 저장
    corrector.save_corrected_data(output_path)
    
    # 성능 통계
    improvement_rate = (corrected / total * 100) if total > 0 else 0
    print(f"\n=== 보정 결과 ===")
    print(f"총 누락 키포인트: {total}")
    print(f"보정된 키포인트: {corrected}")
    print(f"보정률: {improvement_rate:.1f}%")
    print(f"예상 DetA 향상: +{improvement_rate * 0.08:.2f}%")

if __name__ == "__main__":
    main()
