# Back Angle Measurement Model

This is an implementation of the back angle measurement model in the paper. This model takes images as well as videos as input, measuring the back angle in real time without hassle!

## How to use

### back angle detection for images

```bash
python3 code/process_img.py --seg_model segmod_path --detection_model detmod_path --input inp_img_path --output output_path --orientation left/right --background_color white/black
```

### back angle detection for videos

```bash
python3 code/process_vid.py --seg_model segmod_path --detection_model detmod_path --input inp_img_path --output output_path --orientation left/right --background_color white/black
```

## Helper scripts

### image post-grayscale-blurring

```bash
python3 code/grayscale_blur.py --seg_model segmod_path --detection_model detmod_path --input inp_img_path --output output_path --background_color white/black
```

### saving a frame from a video
```bash
python3 code/save_frame_from_vid.py
```