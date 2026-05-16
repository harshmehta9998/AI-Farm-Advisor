# models/saved_model/

Place your trained model files here:

- `disease_model.h5`    — Trained Keras model (run `models/train_model.py` to generate)
- `class_labels.json`   — List of class names in order of model output indices

## How to get a trained model

### Option A: Train your own (recommended for real use)
1. Download PlantVillage dataset from Kaggle
2. Place it at `data/plantvillage/`
3. Run: `python models/train_model.py`

### Option B: Use mock predictions (for prototyping)
If no `.h5` file is present, the app automatically falls back to
random mock predictions so you can still demo the full pipeline.

## class_labels.json format
```json
[
  "Tomato - Bacterial Spot",
  "Tomato - Early Blight",
  "Tomato - Healthy",
  ...
]
```
The index position matches the model's output neuron index.
