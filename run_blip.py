import argparse
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import vqa_utils

def main():
    parser = argparse.ArgumentParser(description="Run BLIP VQA Model")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    data = vqa_utils.load_data(limit=args.limit)
    
    print("Initializing BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    print(f"\nEvaluating BLIP...")
    model_accs = []
    
    for sample in tqdm(data, desc="BLIP"):
        try:
            image = Image.open(sample['image_path']).convert("RGB")
            text = sample['question']
            
            inputs = processor(image, text, return_tensors="pt")
            output = model.generate(**inputs)
            pred = processor.decode(output[0], skip_special_tokens=True)
            
            acc = vqa_utils.compute_vqa_accuracy(pred, sample['answers'])
            model_accs.append(acc)

            if len(model_accs) <= 3:
                    print(f"Q: {text} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")
        except Exception as e:
            print(f"Error processing sample {sample['question_id']}: {e}")
    
    if model_accs:
        avg_acc = sum(model_accs) / len(model_accs)
        print(f"BLIP Average Accuracy: {avg_acc:.4f}")
    else:
        print("No samples evaluated.")

if __name__ == "__main__":
    main()
