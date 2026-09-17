# README Figures

`qemscore-overview.png` is the README's hero image. It is Figure 1 of the QEMScore paper, revision
5, with the three section pointers that index the paper removed, because on GitHub they would
point at nothing.

| File | What it is |
|---|---|
| `QEMScore-figure1-v5.pptx` | The editable source, one slide, native PowerPoint shapes and connectors. SHA-256 `7c9418eb9609d6636e330de63a30c12d3d7036a76042a967434c3f471df03719`. |
| `qemscore-overview.png` | The README image, 2400 by 632 pixels, exported from the PPTX and then passed through the helper below. |
| `strip_section_labels.py` | Removes the section pointers from a PNG export by painting each label's bounding box in its panel's background color. |

To rebuild after editing the PPTX: export the slide from desktop PowerPoint as PNG at 2400 pixels
wide, then run

```bash
python strip_section_labels.py QEMScore-figure1-v5.png qemscore-overview.png
```

The helper needs Pillow. It reports the three boxes it painted; if it reports fewer than three,
the export geometry changed and the panel extents at the top of the script need updating.

The palette is the paper's: green `#365E49` on mint for the control side, coral `#ED8D5A` and
orange `#B66037` on peach for the measurement side, ochre `#A96C21` on gold for the ideal label.
