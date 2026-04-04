from datasets import load_dataset
import os, json

def download_alpaca(): 
    print(f"downloading the alpaca dataset")
    dataset = load_dataset(path="tatsu-lab/alpaca", split="train")

    # shuffling the dataset 
    print(f"shuffling the alpaca dataset")
    dataset = dataset.shuffle(seed=42).select(range(5000))

    # filtering the ones whose characters are less than 10 
    print("filtering outputs with less than 10 characters")
    dataset = dataset.filter(lambda x: len(x["output"]) >= 10)

    # save dataset as json 
    output_path = "data/raw/alpaca_en.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"saving the alpaca dataset to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f: 
        for record in dataset: 
            f.write(
                json.dumps({
                    "instruction": record["instruction"], 
                    "input": record["input"], 
                    "output": record["output"]
                }) + "\n"
            )

    print(f"saved {len(dataset)} samples.")

if __name__ == "__main__": 
    download_alpaca()