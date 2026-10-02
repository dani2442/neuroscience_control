# Presentation assets

Figures and animation frames for `paper_submission/ppt.tex` (beamer, metropolis).

| script | output (under `paper_submission/images/ppt/`) |
|---|---|
| `make_fmri_animation.py` | `frames_fmri/f-000.png` … `f-079.png` — animated "what is fMRI" panel (glass brain + data matrix), built from the Schaefer-100 atlas and the HCP time series. Portrait layout, one frame per TR, played at 8 fps (≈ 5.8 s of brain time per second) |
| `make_analytic_animation.py` | `frames_analytic/f-000.jpg` … `f-124.jpg` — animated analytic signal (real BOLD trace + its Hilbert transform on the left, the point $z(t)$ tracing the complex plane on the right), every other TR, played at 12 fps |
| `make_math_figures.py` | `hopf_bifurcation.pdf`, `hopf_regimes.pdf`, `analytic_signal.pdf`, `grid_cost.pdf`, `ks_surrogate.pdf` |

Data used (already present locally):
- `~/nilearn_data/schaefer_2018/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz`
- `data/ts_young/ts_young_TR0.72.mat` (HCP, 100 ROIs x 1200 TRs x 94 subjects)

`images/ppt/fmri_scanner.jpg` is `images/fmri.avif` converted with ImageMagick
(xelatex cannot read AVIF).

`frames_models/` is extracted from the rotating cortical-surface GIF:

```sh
convert empirical_vs_four_models__rotating.gif -coalesce -resize 1320x \
        -quality 88 frames_models/f-%03d.jpg
```

## Building the deck

```sh
cd paper_submission
latexmk -xelatex ppt.tex      # metropolis needs Fira -> xelatex or lualatex
```

The three `\animategraphics` blocks play only in a viewer that
runs PDF JavaScript — Adobe Acrobat/Reader, pdf-presenter-console, Okular.
In other viewers the first frame is shown as a still (`poster=first`).
