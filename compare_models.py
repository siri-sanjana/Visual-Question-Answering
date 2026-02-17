
import json
import os
import argparse
import random
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

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
    
    qid2annot = {a['question_id']: a for a in annotations}
    
    data = []
    for q in questions:
        qid = q['question_id']
        if qid in qid2annot:
            annot = qid2annot[qid]
            img_id = q['image_id']
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
    pred = prediction.lower().strip()
    
    count = 0
    for ans in truth_answers:
        if ans.lower().strip() == pred:
            count += 1
            
    return min(count / 3.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Compare VQA models")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--models", nargs='+', default=["vilt", "blip", "blip2"], 
                        help="Models to evaluate")
    args = parser.parse_args()
    
    data = load_data(limit=args.limit)
    
    results = defaultdict(list)
    
    # ViLT
    if "vilt" in args.models:
        try:
            from transformers import ViltProcessor, ViltForQuestionAnswering
            print("Initializing ViLT...")
            processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            
            print(f"\nEvaluating ViLT...")
            model_accs = []
            
            for sample in tqdm(data, desc="ViLT"):
                image = Image.open(sample['image_path']).convert("RGB")
                text = sample['question']
                
                # prepare inputs
                encoding = processor(image, text, return_tensors="pt")
                
                # forward pass
                outputs = model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                pred = model.config.id2label[idx]
                
                acc = compute_vqa_accuracy(pred, sample['answers'])
                model_accs.append(acc)
                results["ViLT"].append(acc)
                
                # Optional: print first few predictions
                if len(model_accs) <= 3:
                     print(f"Q: {text} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")

            avg_acc = sum(model_accs) / len(model_accs)
            print(f"ViLT Average Accuracy: {avg_acc:.4f}")
            del model, processor # clean up
        except Exception as e:
            print(f"Error evaluating ViLT: {e}")

    # BLIP
    if "blip" in args.models:
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            print("Initializing BLIP...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            
            print(f"\nEvaluating BLIP...")
            model_accs = []
            
            for sample in tqdm(data, desc="BLIP"):
                image = Image.open(sample['image_path']).convert("RGB")
                text = sample['question']
                
                inputs = processor(image, text, return_tensors="pt")
                output = model.generate(**inputs)
                pred = processor.decode(output[0], skip_special_tokens=True)
                
                acc = compute_vqa_accuracy(pred, sample['answers'])
                model_accs.append(acc)
                results["BLIP"].append(acc)

                if len(model_accs) <= 3:
                     print(f"Q: {text} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")
            
            avg_acc = sum(model_accs) / len(model_accs)
            print(f"BLIP Average Accuracy: {avg_acc:.4f}")
            del model, processor
        except Exception as e:
            print(f"Error evaluating BLIP: {e}")

    # BLIP-2
    if "blip2" in args.models:
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            print("Initializing BLIP-2...")
            # Try use_fast=False to avoid Rust tokenizer errors if cache is problematic
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            model.to(device)
            
            print(f"\nEvaluating BLIP-2...")
            model_accs = []
            
            for sample in tqdm(data, desc="BLIP-2"):
                image = Image.open(sample['image_path']).convert("RGB")
                text = "Question: " + sample['question'] + " Answer:"
                
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                
                # Explicitly pass arguments and set max_length to avoid error "max_length is set to -1"
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    max_length=150 # Set a safe upper bound to ensure max_length is not -1
                )
                pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                acc = compute_vqa_accuracy(pred, sample['answers'])
                model_accs.append(acc)
                results["BLIP-2"].append(acc)

                if len(model_accs) <= 3:
                     print(f"Q: {sample['question']} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")
            
            avg_acc = sum(model_accs) / len(model_accs)
            print(f"BLIP-2 Average Accuracy: {avg_acc:.4f}")
            del model, processor
        except Exception as e:
            print(f"Error evaluating BLIP-2: {e}")

if __name__ == "__main__":
    main()
