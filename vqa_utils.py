import json
import os
import random
from collections import defaultdict
from PIL import Image

# Paths (Adjust if running elsewhere)
VQA_DIR = "/Users/sirisanjana/Documents/6th sem/nlp/project/VQA2"
IMG_DIR = os.path.join(VQA_DIR, "val2014")
QUEST_FILE = os.path.join(VQA_DIR, "v2_OpenEnded_mscoco_val2014_questions.json")
ANNOT_FILE = os.path.join(VQA_DIR, "v2_mscoco_val2014_annotations.json")

def load_data(limit=None):
    print("Loading questions...")
    with open(QUEST_FILE, 'r') as f:
        questions = json.load(f)['questions']
    
    print("Loading annotations...")
    with open(ANNOT_FILE, 'r') as f:
        annotations = json.load(f)['annotations']
    
    # Map annotations by question_id
    qid2annot = {a['question_id']: a for a in annotations}
    
    data = []
    for q in questions:
        qid = q['question_id']
        if qid in qid2annot:
            annot = qid2annot[qid]
            img_id = q['image_id']
            # COCO image filename format
            img_filename = f"COCO_val2014_{img_id:012d}.jpg"
            img_path = os.path.join(IMG_DIR, img_filename)
            
            if os.path.exists(img_path):
                data.append({
                    'question_id': qid,
                    'image_id': img_id,
                    'image_path': img_path,
                    'question': q['question'],
                    'answers': [a['answer'] for a in annot['answers']],
                    'multiple_choice_answer': annot['multiple_choice_answer']
                })
    
    if limit:
        random.shuffle(data)
        data = data[:limit]
        
    print(f"Loaded {len(data)} samples.")
    return data

def compute_vqa_accuracy(prediction, truth_answers):
    # Standard VQA accuracy: min(1, count/3)
    # prediction should be normalized (lowercase etc if needed, but models usually output lower)
    pred = prediction.lower().strip()
    
    # Simple normalization for truth
    count = 0
    for ans in truth_answers:
        if ans.lower().strip() == pred:
            count += 1
            
    return min(count / 3.0, 1.0)
