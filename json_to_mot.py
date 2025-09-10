import os
import json

def postprocess_multiple_frames(base_dir, pid, output_dir) :
    classes = ["0_left_hand", "1_right_hand", "2_scissors", "3_tweezers", "4_needle_holder", "5_needle"]
    output_lines = []

    for class_id, class_name in enumerate(classes) :
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir) :
            print(f"Warning : {class_dir} not found! Skipping this class directory.")
            continue

        files = [f for f in os.listdir(class_dir) 
                if f.startswith(f"results_{pid}_frame_") and f.endswith('.json')]
        files = sorted(files)

        for json_filename in files :
            json_path = os.path.join(class_dir, json_filename)
            
            with open(json_path, 'r') as f :
                data = json.load(f)

            frame_num = json_filename.split('_frame_')[-1].replace('.json', '')

            for instance in data.get('instance_info', []) :
                keypoints_xy = instance.get('keypoints', [])
                keypoint_scores = instance.get('keypoint_scores', [])

                line = [frame_num, class_id, class_id, -1, -1, -1, -1]
                for (xy, score) in zip(keypoints_xy, keypoint_scores) :
                    line.extend([xy[0], xy[1], score])

                output_lines.append(','.join(map(str, line)))

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{pid}_pred.txt")
    with open(output_path, 'w') as f :
        f.write('\n'.join(output_lines))

    print(f"Saved prediction to {output_path}")


def process_all_pids(base_dir, output_dir) :
    first_class_dir = os.path.join(base_dir, "0_left_hand")
    if not os.path.exists(first_class_dir) :
        print(f"Error : {first_class_dir} not found!")
        return

    pids = set()
    for filename in os.listdir(first_class_dir) :
        if filename.startswith("results_") and filename.endswith(".json") :
            pid = filename.split('_')[1]
            pids.add(pid)

    print(f"Found PIDs : {sorted(pids)}")

    for pid in sorted(pids) :
        print(f"Processing PID : {pid}")
        postprocess_multiple_frames(base_dir, pid, output_dir)


# if __name__ == "__main__" :
#     base_dir = "./results"
#     output_dir = "./postprocessed"
    
#     process_all_pids(base_dir, output_dir)
    
#     postprocess_multiple_frames(base_dir, "E66F", output_dir)
#     postprocess_multiple_frames(base_dir, "A66G", output_dir)

base_dir = '/MediSC_OyeSS/results'
output_dir = '/output/trackers'

process_all_pids(base_dir, output_dir)
