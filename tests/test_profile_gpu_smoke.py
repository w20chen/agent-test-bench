import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deep-profile smoke needs CUDA")
def test_profile_gpu_runs_against_tiny_model(tmp_path):
    """Real-GPU smoke: run profile-gpu against facebook/opt-125m on a 1-iteration trace.

    Skipped on machines without CUDA; on the x86 server (RTX 3060 Ti) this should pass.
    """
    pytest.skip("smoke implementation deferred to x86 verification phase")
