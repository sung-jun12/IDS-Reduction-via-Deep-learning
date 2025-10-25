import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Image Quality Assessment
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# -------------------- DNA Coding --------------------
def bytes_to_bits(b: bytes):
    return ''.join(f'{byte:08b}' for byte in b)

def bits_to_bytes(bit_string: str) -> bytes:
    if len(bit_string) % 8 != 0:
        bit_string = bit_string[:len(bit_string) - (len(bit_string) % 8)]
    return bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

def Rinf_encoding(bin_str: str, period_number: int) -> str:
    dic = ['CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT','AA','AC','AG','AT',
           'CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT','AA','AC','AG','AT']
    bin_dic = [f'{i:04b}' for i in range(16)]
    out = []
    L = len(bin_str) // 4
    for i in range(L):
        unit = bin_str[4*i:4*i+4]
        idx = bin_dic.index(unit) - int(i % period_number)
        idx %= len(dic)
        out.append(dic[idx])
    return ''.join(out)

def Rinf_decoding(base_seq: str, period_number: int) -> str:
    units = [base_seq[i:i+2] for i in range(0, len(base_seq), 2)]
    dic = ['CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT','AA','AC','AG','AT']
    bin_dic = [f'{i:04b}' for i in range(16)] * 2
    out = []
    for i,u in enumerate(units):
        if u not in dic:
            out.append('0000')
            continue
        idx = (dic.index(u) + int(i % period_number)) % len(bin_dic)
        out.append(bin_dic[idx])
    return ''.join(out)


# -------------------- IDS Error Injection --------------------
NUC = ['A','C','G','T']

def substitution(seq, ratio, rng):
    n = int(len(seq) * ratio)
    if n <= 0: 
        return seq
    idxs = rng.sample(range(len(seq)), n)
    s = list(seq)
    for i in idxs:
        cur = s[i]
        cand = [x for x in NUC if x != cur]
        s[i] = rng.choice(cand)
    return ''.join(s)

def indel(seq, indel_ratio, rng):
    n_indel = int(round(len(seq) * indel_ratio))
    if n_indel <= 0:
        return seq
    s = list(seq)
    for _ in range(n_indel):
        if len(s) == 0:
            pos = 0
            s[pos:pos] = rng.choice(NUC)
            continue
        if rng.random() < 0.5:
            pos = rng.randrange(0, len(s) + 1)
            s[pos:pos] = rng.choice(NUC)
        else:
            pos = rng.randrange(0, len(s))
            del s[pos]
    return ''.join(s)

def inject_ids_errors(seq, sub_ratio, indel_ratio, rng):
    seq = substitution(seq, sub_ratio, rng)
    seq = indel(seq, indel_ratio, rng)
    return seq


# -------------------- DNA Length Adjustment --------------------
def sequence_length_recovery(seq, original_len):
    if len(seq) < original_len:
        diff = original_len - len(seq)
        pad = ("ACGT" * ((diff // 4) + 1))[:diff]
        return seq + pad, ""
    elif len(seq) > original_len:
        return seq[:original_len], seq[original_len:]
    else:
        return seq, ""


# -------------------- Image I/O --------------------
def load_gray_image(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

def save_gray_image(save_path, arr):
    np_arr = np.asarray(arr, dtype=np.uint8)
    Image.fromarray(np_arr).save(save_path)


# -------------------- Segmentation Model Load --------------------
def segmentation_model_load(model_type, device, weights_path):
    model_type = model_type.lower()
    model = None
    if model_type == "unet":
        from models.unet import UNet
        model = UNet(num_classes=5).to(device)
    elif model_type == "deeplab":
        import torch.nn as nn
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, num_classes=5)
        except Exception:
            from torchvision.models.segmentation import deeplabv3_resnet50
            from torchvision.models.segmentation.deeplabv3 import DeepLabHead
            model = deeplabv3_resnet50(weights=None)
            model.classifier = DeepLabHead(2048, 5)
        model = model.to(device)
    elif model_type == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512" 
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=5,
            ignore_mismatched_sizes=True,
            local_files_only=False).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# -------------------- Denoising Model Load --------------------
def denoising_model_load(device, weights_path):
    from models.dncnn import DnCNN
    net = DnCNN(channels=1, num_layers=17).to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.eval()
    return net


# -------------------- Indel Reduction using Segmentation Model --------------------
def segmentation_indel_reduction(model, model_type, noisy_img, noisy_seq, period=16, device='cpu'):
    import numpy as np
    import torch
    import torch.nn.functional as F

    MAX_ITERS = 50

    H, W = noisy_img.shape
    noisy_img = noisy_img.astype(np.uint8)

    if model is None:
        return noisy_img, noisy_seq

    def run_segmentation(img):
        with torch.no_grad():
            arr = np.ascontiguousarray(img)
            x = torch.from_numpy(arr).to(torch.float32).div(255.0) \
                    .unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
            model.eval()
            out = model(x)
            if model_type == "segformer":
                out = out.logits if hasattr(out, "logits") else out
            elif isinstance(out, dict) and 'out' in out:
                out = out['out']
            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            mask = torch.argmax(out, dim=1).detach().cpu().numpy()[0]
        return mask

    def first_error_pos(mask_hw):
        flat = mask_hw.reshape(-1)
        for i, v in enumerate(flat):
            if v != 0:
                return i, int(v)
        return None, None

    def safe_trim(s, k):
        return s[:-k] if len(s) >= k else s

    def decode_candidate(joined_seq, H, W):
        target_bits = H * W * 8
        target_len  = (target_bits // 4) * 2
        main, _ = sequence_length_recovery(joined_seq, target_len)
        bits     = Rinf_decoding(main, period)
        b        = bits_to_bytes(bits[:target_bits])
        arr      = np.frombuffer(b, dtype=np.uint8).reshape(H, W)
        return arr, main

    curr_img = noisy_img
    curr_seq = noisy_seq

    for _ in range(MAX_ITERS):
        mask_input = run_segmentation(curr_img)
        pos_input, err_type = first_error_pos(mask_input)

        if pos_input is None:
            return curr_img, curr_seq

        dna_chunks = [curr_seq[i:i+4] for i in range(0, len(curr_seq), 4)]
        if len(dna_chunks) == 0 or not (0 <= pos_input < len(dna_chunks)):
            return curr_img, curr_seq
        
        candA = dna_chunks[:]
        candB = dna_chunks[:]

        if err_type == 1:
            candA[pos_input:pos_input+1] = [dna_chunks[pos_input] + 'A']
            candB[pos_input:pos_input+1] = [safe_trim(dna_chunks[pos_input], 3)]
        elif err_type == 2:
            candA[pos_input:pos_input+1] = [dna_chunks[pos_input] + 'AC']
            candB[pos_input:pos_input+1] = [safe_trim(dna_chunks[pos_input], 2)]
        elif err_type == 3:
            candA[pos_input:pos_input+1] = [safe_trim(dna_chunks[pos_input], 1)]
            candB[pos_input:pos_input+1] = [dna_chunks[pos_input] + 'ACG']
        elif err_type == 4:
            candA[pos_input:pos_input+1] = [dna_chunks[pos_input] + 'ACGT']
            candB = dna_chunks[:pos_input] + dna_chunks[pos_input+1:]
        else:
            return curr_img, curr_seq

        try:
            arrA, seqA = decode_candidate(''.join(candA), H, W)
            arrB, seqB = decode_candidate(''.join(candB), H, W)
        except Exception:
            return curr_img, curr_seq

        mask_A = run_segmentation(arrA)
        mask_B = run_segmentation(arrB)

        pos_A, _ = first_error_pos(mask_A)
        pos_B, _ = first_error_pos(mask_B)

        if pos_A is None:
            return arrA, seqA
        if pos_B is None:
            return arrB, seqB

        candidates = [
            ("input", pos_input, curr_img, curr_seq),
            ("A",     pos_A,     arrA,     seqA),
            ("B",     pos_B,     arrB,     seqB),
        ]
        max_pos = max(p for _, p, _, _ in candidates)

        chosen_img, chosen_seq = None, None
        for tag in ("input", "A", "B"):  
            for t, p, img, seq in candidates:
                if t == tag and p == max_pos:
                    chosen_img, chosen_seq = img, seq
                    break
            if chosen_img is not None:
                break

        if chosen_img is curr_img or chosen_seq is curr_seq or max_pos <= pos_input:
            return curr_img, curr_seq

        curr_img, curr_seq = chosen_img, chosen_seq

    return curr_img, curr_seq

# -------------------- Substitution Noise Reduction (Denoising) --------------------
def dncnn_denoise(model, img, device='cuda'):
    if model is None:
        return img.astype(np.uint8)
    with torch.no_grad():
        x = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        out = torch.clamp(x - model(x), 0., 255.).squeeze().cpu().numpy().astype(np.uint8)
        return out


# -------------------- Pipeline --------------------
def process_single_image(input_image, output_dir, encoding_period, sub_ratio, indel_ratio, seg_model_type, device):
    rng = random.Random(os.urandom(16))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    in_path = Path(input_image)
    stem = in_path.stem
    ext = in_path.suffix

    # Image Load
    orig = load_gray_image(str(in_path)).astype(np.uint8)
    H, W = orig.shape

    # DNA Encoding
    orig_bytes = orig.tobytes()
    bits = bytes_to_bits(orig_bytes)
    enc_seq = Rinf_encoding(bits, encoding_period)
    enc_path = os.path.join(output_dir, f"{stem}.dna")
    with open(enc_path, 'w') as f:
        f.write(enc_seq)

    # IDS Error Injection
    noisy_seq = inject_ids_errors(enc_seq, sub_ratio, indel_ratio, rng)
    noisy_path = os.path.join(output_dir, f"{stem}_noisy.dna")
    with open(noisy_path, 'w') as f:
        f.write(noisy_seq)

    # Sequence Length Adjustment with Noisy Image Save
    target_bits = H * W * 8
    target_len  = (target_bits // 4) * 2
    main_noisy,_ = sequence_length_recovery(noisy_seq, target_len)
    noisy_bits = Rinf_decoding(main_noisy, encoding_period)
    noisy_bytes = bits_to_bytes(noisy_bits[:target_bits])
    noisy_img = np.frombuffer(noisy_bytes, dtype=np.uint8).reshape(H, W)
    save_gray_image(os.path.join(output_dir, f"{stem}_noisy{ext}"), noisy_img)

    # Indel-Reduction using Segmentation
    seg_weights = {
        "unet": "./models/unet.pth",
        "deeplab": "./models/deeplab.pth",
        "segformer": "./models/segformer.pth",
    }.get(seg_model_type.lower(), "./models/unet.pth")
    seg_net = segmentation_model_load(seg_model_type, device, seg_weights)
    red_img, red_seq = segmentation_indel_reduction(
        seg_net, seg_model_type.lower(), noisy_img, noisy_seq,
        period=encoding_period, device=device)
    save_gray_image(os.path.join(output_dir, f"{stem}_indel_reduced{ext}"), red_img)

    # DnCNN Denoising
    dncnn = denoising_model_load(device, "./models/dncnn.pth")
    den_img = dncnn_denoise(dncnn, red_img, device=device)
    save_gray_image(os.path.join(output_dir, f"{stem}_denoised{ext}"), den_img)

    # Image Quality Assessment (i.e., PSNR, SSIM)
    Pn, Sn = psnr(orig, noisy_img, data_range=255), ssim(orig, noisy_img, data_range=255)
    Pr, Sr = psnr(orig, red_img, data_range=255), ssim(orig, red_img, data_range=255)
    Pd, Sd = psnr(orig, den_img, data_range=255), ssim(orig, den_img, data_range=255)

    print(f"[Image Quality] Noisy        : PSNR={Pn:.3f}dB, SSIM={Sn:.4f}")
    print(f"[Image Quality] Indel-Reduced: PSNR={Pr:.3f}dB, SSIM={Sr:.4f}")
    print(f"[Image Quality] Denoised     : PSNR={Pd:.3f}dB, SSIM={Sd:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="DNA-encoding → IDS injection → DNA-decoding with indel reduction → DnCNN denoise pipeline")
    # Input/ouput Image Path
    parser.add_argument("--input_image", required=True, type=str, help="Input image path (grayscale)")
    parser.add_argument("--output_dir", default="./data/output_image/", type=str)
    # DNA Coding Scheme
    parser.add_argument("--encoding_period", default=16, type=int)
    # IDS Error Ratio
    parser.add_argument("--sub_ratio", default=0.01, type=float, help="Substitution Ratio")
    parser.add_argument("--indel_ratio", default=0.00003, type=float, help="Indel Ratio")
    # Deep-learning Model
    parser.add_argument("--seg_model", default="unet", choices=["unet","deeplab","segformer"], help="Select Segmentation Model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    process_single_image(
        input_image=args.input_image,
        output_dir=args.output_dir,
        encoding_period=args.encoding_period,
        sub_ratio=args.sub_ratio,
        indel_ratio=args.indel_ratio,
        seg_model_type=args.seg_model,
        device=args.device)

if __name__ == "__main__":
    main()
