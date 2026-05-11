"""
Ultra-compact CJK encoding: 1 char per object, 2 chars total for 2 objects.
idx = angle_deg * 24 + label_idx * 4 + urg_idx  (max = 359*24+5*4+3 = 8639)
Char = VOCAB[idx]
"""
import argparse, json, random, time
from pathlib import Path
from transformers import AutoTokenizer

# Build vocab of 8640 single-token CJK chars
def build_vocab(tok_path: str) -> list[str]:
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab = []
    for rng in [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]:
        for cp in range(rng[0], rng[1]+1):
            c = chr(cp)
            if len(tok.encode(c, add_special_tokens=False)) == 1:
                vocab.append(c)
            if len(vocab) >= 8640:
                return vocab
    return vocab

LABELS = ["car","truck","bus","person","bike","obstacle"]
URGENCIES = ["crit","high","med","low"]
LABEL_IDX = {l:i for i,l in enumerate(LABELS)}
URG_IDX   = {u:i for i,u in enumerate(URGENCIES)}

def encode_obj(vocab, angle_deg, label, urg):
    a = int(round(angle_deg)) % 360
    idx = a * 24 + LABEL_IDX[label] * 4 + URG_IDX[urg]
    return vocab[idx]

def decode_obj(vocab, char):
    idx = vocab.index(char)
    urg_i   = idx % 4
    idx //= 4
    label_i = idx % 6
    angle   = idx // 6
    return angle, LABELS[label_i], URGENCIES[urg_i]

def dist_to_urg(d):
    if d < 2: return "crit"
    if d < 4: return "high"
    if d < 8: return "med"
    return "low"

def random_scene(vocab):
    heading = random.randint(0, 359)
    objs = [(random.choice(LABELS), random.randint(0,359), round(random.uniform(0.5,25),1))
            for _ in range(random.randint(2,5))]
    objs.sort(key=lambda x: x[2])
    p, s = objs[0], objs[1]
    parts = [f"{l}@{d}° {m}m" for l,d,m in objs]
    user = f"heading={heading}° | visual=[" + ", ".join(parts) + "]"
    answer = encode_obj(vocab, p[1], p[0], dist_to_urg(p[2])) + \
             encode_obj(vocab, s[1], s[0], dist_to_urg(s[2]))
    return user, answer

SYSTEM = (
    "Output 2 CJK chars. Each char encodes 1 object (angle+label+urgency). "
    "First char = closest object, second = next closest. No other text."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="train_ultra.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tok", default="/Users/ishaangubbala/models/gemma-4-E2B-it")
    args = ap.parse_args()
    random.seed(args.seed)
    print("Building vocab...")
    vocab = build_vocab(args.tok)
    print(f"Vocab size: {len(vocab)} single-token chars")
    t0 = time.monotonic()
    with open(args.out, "w") as f:
        for i in range(args.n):
            user, gold = random_scene(vocab)
            f.write(json.dumps({"messages":[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":user},
                {"role":"assistant","content":gold},
            ]}, ensure_ascii=False) + "\n")
            if (i+1) % 1000 == 0:
                print(f"  {i+1}/{args.n}")
    print(f"Done {args.n} in {time.monotonic()-t0:.1f}s → {args.out}")
    # Save vocab for decoder
    Path("ultra_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    print("Saved ultra_vocab.json")

if __name__ == "__main__":
    main()
