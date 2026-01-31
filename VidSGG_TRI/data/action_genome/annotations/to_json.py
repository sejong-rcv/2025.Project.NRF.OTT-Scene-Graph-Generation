import pickle
import json
import os
import numpy as np


OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 1. read object_bbox_and_relationship.pkl and save to JSON
# with open('./object_bbox_and_relationship.pkl', 'rb') as f:
#     object_data = pickle.load(f)


# with open(os.path.join(OUTPUT_DIR, 'object_bbox_and_relationship.json'), 'w') as f_json:
#     json.dump(object_data, f_json, indent=4)

# print(" object_bbox_and_relationship.json saved")


def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

with open('./person_bbox.pkl', 'rb') as f:
    person_data = pickle.load(f)

person_data_serializable = convert_to_json_serializable(person_data)


with open(os.path.join(OUTPUT_DIR, 'person_bbox.json'), 'w') as f_json:
    json.dump(person_data_serializable, f_json, indent=4)

print(" person_bbox.json saved")
