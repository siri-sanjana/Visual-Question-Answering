import argparse
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import vqa_utils

def main():
    parser = argparse.ArgumentParser(description="Run BLIP-2 VQA Model")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    data = vqa_utils.load_data(limit=args.limit)
    
    print("Initializing BLIP-2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    model.to(device)
    
    print(f"\nEvaluating BLIP-2...")
    model_accs = []
    
    for sample in tqdm(data, desc="BLIP-2"):
        try:
            image = Image.open(sample['image_path']).convert("RGB")
            text = "Question: " + sample['question'] + " Answer:"
            
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                max_length=150 
            )
            pred_full = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if "Answer:" in pred_full:
                pred = pred_full.split("Answer:")[-1].strip()
            else:
                pred = pred_full
            
            acc = vqa_utils.compute_vqa_accuracy(pred, sample['answers'])
            model_accs.append(acc)

            if len(model_accs) <= 3:
                    print(f"Q: {sample['question']} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")
        except Exception as e:
            print(f"Error processing sample {sample['question_id']}: {e}")
    
    if model_accs:
        avg_acc = sum(model_accs) / len(model_accs)
        print(f"BLIP-2 Average Accuracy: {avg_acc:.4f}")
    else:
        print("No samples evaluated.")

if __name__ == "__main__":
    main()
