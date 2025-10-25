import os
import glob
import argparse
import random

from typing import Tuple, Optional
import numpy as np
from PIL import Image

from main import (Rinf_encoding, Rinf_decoding, sequence_length_recovery, bytes_to_bits, bits_to_bytes)


# ---------------------- Segmentation Map Generation ----------------------
def segmap_generator(image, DNA_Seq, error_position, error_type, HW: Tuple[int, int]):

    H, W = HW
    target_pixels = H * W
    pos2type = {p: t for p, t in zip(error_position, error_type)}

    string_list = list(DNA_Seq)
    new = []
    for i, s in enumerate(string_list):
        if i in pos2type:
            t = pos2type[i]
            if t == 'I':
                new += ['I', s]
            elif t == 'D':
                new += ['D']
            else:
                new += [s]
        else:
            new += [s]

    # Neighbouring Insertion-Deletion is equal to single Substitution!
    i = 0
    while i < len(new) - 1:
        if (new[i], new[i + 1]) in [('D', 'I'), ('I', 'D')]:
            new[i:i + 2] = ['S']
        else:
            i += 1

    # DNA Sequence Reconstruction
    nuc = ['A', 'C', 'G', 'T']
    dna_new = []
    for c in new:
        if c == 'I' or c == 'S':
            dna_new.append(random.choice(nuc))
        elif c != 'D':
            dna_new.append(c)
    mutated_DNA = ''.join(dna_new)

    # Segmantation Map Generation
    cum, chunks, err = 0, [], []
    for c in new:
        chunks.append(c)
        if sum(x != 'D' for x in chunks) == 4:
            ins = chunks.count('I') + chunks.count('S')
            dele = chunks.count('D')
            cum += (ins - dele)
            err.append(cum)
            chunks = []
            if len(err) >= target_pixels:
                break
    if len(err) < target_pixels:
        err += [err[-1] if err else 0] * (target_pixels - len(err))

    seg = np.zeros(target_pixels, np.uint8)
    for i, s in enumerate(err):
        if s == 0:
            v = 0
        elif s > 0:
            v = 4 if s % 4 == 0 else 4 - (s % 4)
        else:
            v = (-s) % 4 or 4
        seg[i] = v
    seg = seg.reshape(H, W)
    return mutated_DNA, seg


# ---------------------- Utilities ----------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_force_size(s):
    if not s:
        return None
    w, h = map(int, s.lower().replace('x', ' ').split())
    return (h, w)


def load_gray_uint8(p, force_hw=None):
    im = Image.open(p).convert('L')
    if force_hw:
        H, W = force_hw
        if im.size != (W, H):
            im = im.resize((W, H), Image.NEAREST)
    return np.array(im, np.uint8)


def save_gray_uint8(p, arr):
    Image.fromarray(arr, 'L').save(p)


def colorize_segmap(seg_uint8: np.ndarray) -> np.ndarray:
    color_map = {
        0: (128, 255, 128),  ## green : Correct pixel
        1: (255, 200, 128),  ## yellow : Type I Error (aligned pattern)
        2: (128, 128, 255),  ## blue : Type II Error (clustered pattern)
        3: (255, 128, 128),  ## red : Type III Error (scattered pattern)
        4: (200, 128, 255),  ## purple : Type IV Error (preserved pattern)
    }
    h, w = seg_uint8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in color_map.items():
        rgb[seg_uint8 == val] = color
    return rgb


# ---------------------- IDS Noise Adoption ----------------------
def substitution(seq, ratio):

    NUC = "ACGT"
    L = len(seq)
    if L == 0 or ratio <= 0:
        return seq

    n = int(round(L * ratio))
    n = max(0, min(L, n))
    if n == 0:
        return seq

    idxs = random.sample(range(L), n)
    s = list(seq)
    for i in idxs:
        cur = s[i]
        cand = [x for x in NUC if x != cur] if cur in NUC else list(NUC)
        s[i] = random.choice(cand)
    return ''.join(s)


def IDS_segmap_pipeline(img_uint8, period, sub_ratio, indel_ratio):

    H, W = img_uint8.shape[:2]
    need_bits = H * W * 8

    bits = bytes_to_bits(img_uint8.reshape(-1))
    dna = Rinf_encoding(bits, period)

    # IDS Error Injection (Substitution -> Indel)
    dna = substitution(dna, sub_ratio)

    original_len = len(dna)

    if indel_ratio <= 0:
        mutated_dna = dna
        seg = np.zeros((H, W), np.uint8)
    else:
        L = len(dna)
        nerr = int(round(L * indel_ratio)) 
        nerr = max(0, min(L, nerr))
        if nerr == 0:
            mutated_dna = dna
            seg = np.zeros((H, W), np.uint8)
        else:
            pos = sorted(random.sample(range(L), nerr))
            types = [random.choice(['I', 'D']) for _ in range(nerr)]
            mutated_dna, seg = segmap_generator(img_uint8, dna, pos, types, (H, W)) ## noisy dna, segmenation map generation

    # Sequence length recover (due to indel noises)
    main_dna, *_ = sequence_length_recovery(mutated_dna, original_len)

    binary = Rinf_decoding(main_dna, period)
    if len(binary) < need_bits:
        binary += [0] * (need_bits - len(binary))
    noisy = np.frombuffer(bytearray(bits_to_bytes(binary[:need_bits])), np.uint8).reshape(H, W) ## Noisy image generation
    return mutated_dna, noisy, seg


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default='./data/train_image')
    ap.add_argument('--out_dna_dir', default='./data/train_noisy_dna')
    ap.add_argument('--out_noise_dir', default='./data/train_noisy')
    ap.add_argument('--out_segmap_dir', default='./data/train_segmap')

    ap.add_argument('--encoding_period', type=int, default=16) ## Rinf-P16 Coding
    ap.add_argument('--sub_ratio', type=float, default=0.005)
    ap.add_argument('--indel_ratio', type=float, default=0.0001)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--ext', default='png')
    ap.add_argument('--force_size', default=None) ## Image resizing 
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for d in [args.out_dna_dir, args.out_noise_dir, args.out_segmap_dir]:
        ensure_dir(d)

    force_hw = parse_force_size(args.force_size)
    files = sorted(glob.glob(os.path.join(args.input_dir, f'*.{args.ext}')))

    for i, fp in enumerate(files, 1):
        try:
            img = load_gray_uint8(fp, force_hw)
            dna, noisy, seg = IDS_segmap_pipeline(
                img, args.encoding_period, args.sub_ratio, args.indel_ratio
            )
            base = os.path.splitext(os.path.basename(fp))[0]

            dna_path  = os.path.join(args.out_dna_dir,  f'{base}.dna')
            noisy_path= os.path.join(args.out_noise_dir,f'{base}.png')
            seg_path  = os.path.join(args.out_segmap_dir,f'{base}.png')

            with open(dna_path, 'w') as f:
                f.write(dna)
            save_gray_uint8(noisy_path, noisy)
            seg_rgb = colorize_segmap(seg)
            Image.fromarray(seg_rgb, mode='RGB').save(seg_path)  ## Visualization for users (must be converted back to segmantation map with 0,1,2,3,and 4 when training)

            if i % 50 == 0:
                print(f'[{i}/{len(files)}] done')

        except Exception as e:
            print(f'[WARN] {fp}: {e}')

    print(f'Finished!: DNA File->{args.out_dna_dir}, Noisy Image->{args.out_noise_dir}, Segmentation Maps->{args.out_segmap_dir}')


if __name__ == '__main__':
    main()
