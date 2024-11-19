# G2D-Boosting-Multimodal-Learning-with-Gradient-Guided-Distillation

This is the official PyTorch implementation of the paper G2D.


## Training

### Environment config

1. Python: 3.11.0
2. CUDA Version: 11.7
3. Pytorch: 2.0.1
4. Torchvision: 0.15.2

### Train

1. Set the hyperparameters in ``config.py``.
2. Train the teacher using the teacher.py script:

```python
python teacher.py
```

3. Train the student using the student.py script:

```python
python student.py
```
