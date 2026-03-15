import argparse
import json
from collections import OrderedDict


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_video_info(records):
    video2dur = OrderedDict()
    for e in records:
        vid = e.get("video") or e.get("vid_name")
        dur = e.get("duration")
        if vid is None or dur is None:
            raise ValueError("Missing video/duration in {}".format(e))
        if vid not in video2dur:
            video2dur[vid] = float(dur)
    return video2dur


def build_index(video2dur):
    video2idx = OrderedDict()
    for idx, vid in enumerate(video2dur.keys()):
        video2idx[vid] = [video2dur[vid], idx]
    return video2idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train", type=str)
    parser.add_argument("--val", type=str)
    parser.add_argument("--val_1", type=str)
    parser.add_argument("--val_2", type=str)
    parser.add_argument("--test", type=str)
    args = parser.parse_args()

    split_paths = OrderedDict()
    for key in ["train", "val", "val_1", "val_2", "test"]:
        path = getattr(args, key)
        if path:
            split_paths[key] = path

    if not split_paths:
        raise ValueError("No split files provided.")

    output = OrderedDict()
    for split, path in split_paths.items():
        records = load_jsonl(path)
        video2dur = extract_video_info(records)
        output[split] = build_index(video2dur)

    with open(args.output, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
