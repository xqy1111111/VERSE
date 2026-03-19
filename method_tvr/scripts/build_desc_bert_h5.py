import argparse
import json

import h5py
import torch
from transformers import AutoModel, AutoTokenizer


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_h5", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="fig_desc")
    parser.add_argument("--id_key", type=str, default="desc_id")
    parser.add_argument("--tokenizer", type=str, default="roberta-base")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    model = AutoModel.from_pretrained(args.tokenizer).to(device)
    model.eval()

    data = load_jsonl(args.input_jsonl)
    with h5py.File(args.output_h5, "w") as h5f:
        with torch.no_grad():
            for e in data:
                desc_id = str(e[args.id_key])
                text = e[args.text_key]
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded)
                feat = output.last_hidden_state.squeeze(0).cpu().numpy()
                h5f.create_dataset(desc_id, data=feat)


if __name__ == "__main__":
    main()
