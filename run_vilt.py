import argparse
from tqdm import tqdm
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import vqa_utils

def main():
    parser = argparse.ArgumentParser(description="Run ViLT VQA Model")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    data = vqa_utils.load_data(limit=args.limit)
    
    print("Initializing ViLT...")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
    print(f"\nEvaluating ViLT...")
    model_accs = []
    
    for sample in tqdm(data, desc="ViLT"):
        try:
            image = Image.open(sample['image_path']).convert("RGB")
            text = sample['question']
            
            encoding = processor(image, text, return_tensors="pt")
            
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            pred = model.config.id2label[idx]
            
            acc = vqa_utils.compute_vqa_accuracy(pred, sample['answers'])
            model_accs.append(acc)
            
            if len(model_accs) <= 3:
                    print(f"Q: {text} | Pred: {pred} | Truth: {sample['multiple_choice_answer']}")
        except Exception as e:
            print(f"Error processing sample {sample['question_id']}: {e}")

    if model_accs:
        avg_acc = sum(model_accs) / len(model_accs)
        print(f"ViLT Average Accuracy: {avg_acc:.4f}")
    else:
        print("No samples evaluated.")

if __name__ == "__main__":
    main()
