from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json, os

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to("cuda")

def translate_text(
    model, 
    tokenizer,
    text: str, 
): 
    """translate text from english (eng_Ltn) to yoruba (yor_Ltn)"""
    if not text or text.strip() == "": 
        return ""
    
    # set source language to english 
    tokenizer.src_lang = "eng_Latn"

    # tokenize the text 
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # translate tokens 
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("yor_Latn")
    )

    # decode back to string 
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

def translate_dataset(
    input_path: str, 
    output_path: str
): 
    translated_count = 0
    resume_from = 0 

    if os.path.exists(output_path): 
        with open(output_path, "r") as f: 
            resume_from = sum(1 for line in f)
        print(f"resuming translation from line: {resume_from}")


    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "a", encoding="utf-8") as f_out: 
        for i, line in enumerate(f_in): 
            if i < resume_from: 
                continue
            try: 
                record = json.loads(line)

                instruction_en = record.get("instruction", "")
                input_en = record.get("input", "")
                output_en = record.get("output", "")

                instruction_yor = translate_text(model=model, tokenizer=tokenizer, text=instruction_en)
                input_yor = translate_text(model=model, tokenizer=tokenizer, text=input_en)
                output_yor = translate_text(model=model, tokenizer=tokenizer, text=output_en)

                # construct a new record
                new_record = {
                    "instruction_en": instruction_en, 
                    "input_en": input_en, 
                    "output_en": output_en, 
                    "instruction_yor": instruction_yor, 
                    "input_yor": input_yor, 
                    "output_yor": output_yor
                }

                f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                translated_count += 1 

            except Exception as e: 
                print(f"skipping a record due to exception: {e}")
                continue

        print(f"translation complete | total records translate: {translated_count}")
        



if __name__ == "__main__": 
    translate_dataset(input_path="data/raw/alpaca_en.jsonl", output_path="data/translated/alpaca_yor.jsonl")