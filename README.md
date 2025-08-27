# G<sup>2</sup>D-Boosting-Multimodal-Learning-with-Gradient-Guided-Distillation

This is the official PyTorch implementation of the paper G<sup>2</sup>D.

**Accepted by: ICCV 2025**

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

## Citation
```
@misc{rakib2025g2dboostingmultimodallearning,
      title={G$^{2}$D: Boosting Multimodal Learning with Gradient-Guided Distillation}, 
      author={Mohammed Rakib and Arunkumar Bagavathi},
      year={2025},
      eprint={2506.21514},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.21514}, 
}
