# livingpark-entropy

## Running

This section details how to vizualize segmentations that were processed using the Freesurfer pipeline.

### Step 1: Input File
Create a json file that defines the dataset to load up following this format:

```json
{
  "111": "MIRIAD_189",
  "112": "MIRIAD_196",
  "113": "MIRIAD_208",
  "114": "MIRIAD_217",
  "115": "MIRIAD_221",
  "116": "MIRIAD_246",
  "117": "MIRIAD_251"
}
```

### Step 2: Generate PNG

Run `python make_gif_seg_only.py --input-dir <data dir> --input <json_data>` if you only want to vizualize the segmentations or `python make_gif_compare.py --input-dir <data dir> --input <json_data>` if you want to superimpose the segmentation over the original MRI.

### Step 3: Start Jekyll

Run `python make_gifs_json.py` to create a reference JSON that points to the appropriate PNG for the static website.

Run `bundle exec jekyll serve`

