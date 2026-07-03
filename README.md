# Football Jersey Color Classifier

A simple computer vision project that classifies football team images (Chelsea vs Manchester) based on the dominant jersey color.

## Method

1. Read image
2. Convert BGR → RGB
3. Compute average RGB color
4. Measure Euclidean distance to Red and Blue
5. Assign the nearest class

## Evaluation

- Confusion Matrix
- Accuracy
- Precision
- Recall

## Run

```bash
pip install -r requirements.txt
python main.py
```
