import pytest
import torch
from mixtral_lab.train_utils import init_distributed


def test_force_cpu_even_if_gpu_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rank, world, device = init_distributed("single", "cpu")
    assert (rank, world, device.type) == (0, 1, "cpu")


def test_invalid_device():
    with pytest.raises(ValueError):
        init_distributed("single", "does-not-exist")


def test_no_cuda_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA"):
        init_distributed("single", "cuda")
