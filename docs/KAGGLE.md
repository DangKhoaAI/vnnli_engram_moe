# Kaggle Usage

Use [notebooks/kaggle_wrapup.ipynb](/home/khoa/KHOA/FPTStudy/Semester7/DAT301m/PROJECT/train_model/notebooks/kaggle_wrapup.ipynb) as the uploadable notebook.

## Workflow

1. Upload the notebook to Kaggle.
2. Set `REPO_URL` if needed.
3. Either:
   - set `DATA_DIR` to the folder containing `train.jsonl`, `dev.jsonl`, and `test.jsonl`, or
   - use `--hf-dataset uitnlp/ViANLI` in the train command to load directly from Hugging Face.
4. Choose `MODEL_CONFIG`.
5. Toggle `RUN_FULL_TRAIN`.
6. Run the notebook top-to-bottom.

## Output Location

The notebook writes artifacts under:

```text
/kaggle/working/outputs
```

## Common Adjustments

- Change to another baseline by swapping `MODEL_CONFIG`.
- Run a quick smoke pass by leaving `RUN_FULL_TRAIN = False`.
- Enable full training when the Kaggle session already has the needed runtime and data mounted.

## Troubleshooting

- If `uv` is missing, the notebook installs it first.
- If cloning into `/kaggle/working` fails, check internet settings in the Kaggle notebook session.
- If PhoBERT segmentation is needed, install the segmenter package in an extra setup cell before training.
