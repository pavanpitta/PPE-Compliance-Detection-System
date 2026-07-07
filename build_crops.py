import os, glob, json, collections
import PIL.Image as Image

# YOLO ids: 0 Hardhat, 2 NO-Hardhat, 4 NO-Safety Vest, 5 Person, 7 Safety Vest
base = 'dataset/css-data'
OUT = 'crops'
os.makedirs(OUT, exist_ok=True)

def parse(lbl):
    b = []
    for line in open(lbl):
        p = line.split()
        if len(p) == 5:
            b.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return b

def inside(a, b):
    # center of attribute box a falls within person box b
    return abs(a[1] - b[1]) < b[3] / 2 and abs(a[2] - b[2]) < b[4] / 2

rows = []
cnt = collections.Counter()
for split in ['train', 'valid', 'test']:
    for lbl in glob.glob(f'{base}/{split}/labels/*.txt'):
        boxes = parse(lbl)
        persons = [b for b in boxes if b[0] == 5]
        img = lbl.replace('/labels/', '/images/').replace('.txt', '.jpg')
        if not os.path.exists(img) or not persons:
            continue
        im = Image.open(img).convert('RGB')
        W, H = im.size
        for k, pb in enumerate(persons):
            hat = any(b[0] == 0 and inside(b, pb) for b in boxes)
            nohat = any(b[0] == 2 and inside(b, pb) for b in boxes)
            vest = any(b[0] == 7 and inside(b, pb) for b in boxes)
            novest = any(b[0] == 4 and inside(b, pb) for b in boxes)
            H_ = hat if (hat and nohat) else (hat and not nohat)
            V_ = vest if (vest and novest) else (vest and not novest)
            if H_ and V_:
                c = 'PPE_Compliant'
            elif H_ and not V_:
                c = 'HardHat_Only'
            elif V_ and not H_:
                c = 'Vest_Only'
            elif (nohat or novest) and not H_ and not V_:
                c = 'NO_PPE'
            else:
                continue
            cx, cy, bw, bh = pb[1], pb[2], pb[3], pb[4]
            x1 = max(0, int((cx - bw / 2) * W)); y1 = max(0, int((cy - bh / 2) * H))
            x2 = min(W, int((cx + bw / 2) * W)); y2 = min(H, int((cy + bh / 2) * H))
            if x2 - x1 < 20 or y2 - y1 < 40:
                continue
            crop = im.crop((x1, y1, x2, y2)).resize((96, 96))
            fn = f'{OUT}/{split}_{os.path.basename(lbl)[:-4]}_{k}.jpg'
            crop.save(fn, quality=90)
            rows.append((fn, c, split)); cnt[c] += 1

json.dump(rows, open('crop_labels.json', 'w'))
print('crops:', len(rows))
print(cnt)
