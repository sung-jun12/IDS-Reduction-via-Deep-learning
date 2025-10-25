
# Insertion and deletion (indel) error reduction using deep-learning segmentation for DNA data storage

This project implements a pipeline for predicting and reducing indel (insertion–deletion) errors in DNA-encoded image sequences using deep-learning–based segmentation models, alongside substitution-oriented noise reduction through image denoising. Three segmentation networks—UNet, DeepLabv3, and SegFormer [1–3]—are evaluated for their ability to detect indel positions from generated segmentation maps, while a DnCNN [4] model is employed to suppress substitution noise in the decoded images.

Input images are grayscale samples derived from the STL-10 dataset located in "data/input/". The complete IDS (i.e., insertion, deletion, and substitution) error–reduction workflow proceeds as follows:
Input grayscale image → Byte-to-binary conversion → DNA encoding (binary-to-DNA sequence) → IDS noise injection → DNA decoding with segmentation-based indel reduction (noisy image → segmentation → indel correction) → DnCNN denoising.

[References]
[1] Ronneberger, O., Fischer, P. & Brox, T. U-Net: convolutional networks for biomedical image segmentation. In Lect. Notes Comput. Sci. 9351, 234–241 (2015).
[2] Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F. & Adam, H. Encoder–decoder with atrous separable convolution for semantic image segmentation. In Proc. Eur. Conf. Comput. Vis. 801–818 (2018).
[3] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M. & Luo, P. SegFormer: simple and efficient design for semantic segmentation with transformers. Adv. Neural Inf. Process. Syst. 34, 12077–12090 (2021).
[4] Zhang, K., Zuo, W., Chen, Y., Meng, D. & Zhang, L. Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising. IEEE Trans. Image Process. 26, 3142–3155 (2017).


## Install Requirements

pip install -r requirements.txt


## Running 

python main.py \
  --input_image ./data/input/01.png \
  --output_dir ./data/output/ \
  --encoding_period 16 \
  --sub_ratio 0.005 \
  --indel_ratio 0.00003 \
  --seg_model unet \
  --device cuda


## Outputs

Encoded DNA sequence: ./data/output_image/01.dna
IDS-noisy DNA sequence: ./data/output_image/01_noisy.dna
Noisy image (before indel reduction): ./data/output_image/01_noisy.png
Indel-reduced image (segmentation-guided): ./data/output_image/01_indel_reduced.png
Denoised image (DnCNN): ./data/output_image/01_denoised.png


## References

[1] Ronneberger, O., Fischer, P., Brox, T. U-Net. LNCS 9351, 234–241 (2015).
[2] Chen, L.-C., et al. Encoder–Decoder with Atrous Separable Convolution. ECCV (2018).
[3] Xie, E., et al. SegFormer. NeurIPS 34, 12077–12090 (2021).
[4] Zhang, K., et al. DnCNN. IEEE TIP 26, 3142–3155 (2017).


## Notes

The implementation uses Rinf_encoding/decoding (4-bit to dinucleotide mapping with periodic shift).
Segmentation weights are loaded from ./models/*.pth; SegFormer may download base weights on first run (see requirements).
STL-10 copyrights and model weight licenses are separate from this repository’s license.


## Cite
