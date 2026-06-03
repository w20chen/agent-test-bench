import torch
from torch import nn

from harness.component_memory_profiler import (
    ATTN_KIND,
    MLP_KIND,
    attach_component_hooks,
)


class FakeAttention(nn.Module):
    def forward(self, x):
        return x


class FakeMLP(nn.Module):
    def forward(self, x):
        return x


class FakeFFN(nn.Module):
    def forward(self, x):
        return x


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = FakeMLP()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoder() for _ in range(2)])
        self.unrelated = nn.Linear(8, 8)


def test_hook_identification_finds_attn_and_mlp():
    model = FakeModel()
    profiler = attach_component_hooks(model)
    x = torch.randn(1, 4)
    for layer in model.layers:
        layer.attn(x)
        layer.mlp(x)
    profiler.record_step()
    profiler.detach()
    step = profiler.steps[0]
    # 2 attn + 2 mlp expected
    assert sum(1 for r in step.per_module if r["kind"] == ATTN_KIND) == 2
    assert sum(1 for r in step.per_module if r["kind"] == MLP_KIND) == 2
    # Linear layer NOT included
    assert all("Linear" not in r["module_class"] for r in step.per_module)


def test_ffn_pattern_also_matched():
    model = nn.Sequential()
    model.add_module("ffn", FakeFFN())
    profiler = attach_component_hooks(model)
    model.ffn(torch.randn(1, 4))
    profiler.record_step()
    profiler.detach()
    step = profiler.steps[0]
    assert any(r["kind"] == MLP_KIND for r in step.per_module)


def test_no_attn_no_mlp_logs_warning(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    profiler = attach_component_hooks(model)
    profiler.detach()
    assert any("no attn/mlp submodules" in m for m in caplog.messages)


def test_custom_measurement_fn_called():
    calls = {"n": 0}

    def fake_measure(module, inputs, output, pre):
        calls["n"] += 1
        return {"value_mib": 7.5, "measurement_kind": "test_const"}

    model = nn.Sequential()
    model.add_module("attn", FakeAttention())
    profiler = attach_component_hooks(model, measurement_fn=fake_measure)
    model.attn(torch.randn(1, 4))
    profiler.record_step()
    profiler.detach()
    assert calls["n"] == 1
    assert profiler.steps[0].attn_mib == 7.5
    assert profiler.steps[0].measurement_kind == "test_const"


def test_record_step_increments_index_and_resets_buffer():
    model = nn.Sequential()
    model.add_module("attn", FakeAttention())
    profiler = attach_component_hooks(model)
    model.attn(torch.randn(1, 4))
    profiler.record_step()
    model.attn(torch.randn(1, 4))
    profiler.record_step()
    profiler.detach()
    assert len(profiler.steps) == 2
    assert profiler.steps[0].step_index == 0
    assert profiler.steps[1].step_index == 1


def test_user_todo_marker_present():
    """Sanity guard: the user-contribution placeholder must remain visible
    in source so contributors find it. Stops the marker from being
    accidentally deleted by future refactors."""
    import inspect

    from harness import component_memory_profiler

    source = inspect.getsource(component_memory_profiler)
    assert "# TODO(user):" in source
