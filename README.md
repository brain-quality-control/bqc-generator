# Brain Quality Control Generator

## Input dir structure

The input directory should contain the segmentation files organized in a way that each file path includes at least one occurrence of the string `rep<repetition>`, where `<repetition>` is a unique identifier for each repetition. This repetition identifier is extracted from the file path and used to generate the corresponding GIF. For example, a valid file path might look like:

```
/data/subject1/rep1/segmentation.nii.gz
```

Here, `rep1` is the repetition identifier. The script will search for such patterns in the directory structure to process and generate GIFs for each repetition.

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

Run `python3 scripts/make_gifs.py --input-dir <data dir> --input <json_data>` to generate GIFs and PNGs.

### Step 3: Start Jekyll

#### Generate referencement

Run `python3 scripts/make_gifs_json.py` to create a reference JSON that points to the appropriate PNG for the static website.

Run `bundle exec jekyll serve`

