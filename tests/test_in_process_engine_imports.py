def test_in_process_engine_module_imports_without_vllm():
    """The module itself must import even when vllm isn't installed
    (lazy import inside __init__)."""
    import serving.in_process_engine as m

    assert hasattr(m, "InProcessEngine")
    # Constructor will fail because vllm is not installed; that's fine.


def test_in_process_engine_constructor_raises_clean_error_on_missing_vllm():
    import pytest

    from serving.in_process_engine import InProcessEngine

    # If vllm IS installed in the test env, skip; otherwise expect ImportError
    try:
        import vllm  # noqa: F401
        pytest.skip("vllm is installed; cannot test missing-vllm error path")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="deep-profile"):
        InProcessEngine(model="foo")


def test_in_process_engine_rejects_tensor_parallel_gt_1():
    import pytest

    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vllm not installed; cannot exercise the runtime path")
    from serving.in_process_engine import InProcessEngine

    with pytest.raises(NotImplementedError, match="tensor_parallel_size=1"):
        InProcessEngine(model="x", tensor_parallel_size=2)
