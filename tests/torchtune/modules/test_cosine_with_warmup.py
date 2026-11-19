# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

import torch
import torch.optim as optim

from tests.test_utils import assert_expected

from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup


class TestCosineLR:
    @pytest.fixture
    def scheduler(self):
        optimizer = optim.SGD([torch.ones(1)], lr=0.2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            num_cycles=1.0,
        )
        return scheduler

    def test_cosine_schedule_init(self, scheduler):
        optimizer = scheduler.optimizer
        assert_expected(optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_schedule_mid_warmup(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 5 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.1)

    def test_cosine_schedule_warmup(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 10 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.2)

    def test_cosine_schedule_minimum_value(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 55 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_schedule_complete_cycle(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 100 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.2)

    @pytest.mark.parametrize("optimizer_cls", [optim.SGD, optim.AdamW])
    @pytest.mark.parametrize("closure_mode", ["none", "positional", "keyword"])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_cosine_schedule_optimizer(self, optimizer_cls, closure_mode, compiled):
        model = torch.nn.Linear(2, 1, dtype=torch.float64)
        reference = torch.nn.Linear(2, 1, dtype=torch.float64)
        reference.load_state_dict(model.state_dict())
        optimizer = optimizer_cls(model.parameters(), lr=0.2)
        reference_optimizer = optimizer_cls(reference.parameters(), lr=0.2)
        if compiled:
            optimizer.step = torch.compile(optimizer.step, backend="eager")
            reference_optimizer.step = torch.compile(
                reference_optimizer.step, backend="eager"
            )
        scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 4)
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)

        for _ in range(4):
            reference_optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
            for module, opt in (
                (model, optimizer),
                (reference, reference_optimizer),
            ):

                def closure():
                    opt.zero_grad()
                    loss = module(inputs).square().mean()
                    loss.backward()
                    return loss

                if closure_mode == "positional":
                    loss = opt.step(closure)
                    assert loss.ndim == 0
                elif closure_mode == "keyword":
                    loss = opt.step(closure=closure)
                    assert loss.ndim == 0
                else:
                    closure()
                    opt.step()
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(
                    optimizer.state[parameter],
                    reference_optimizer.state[expected],
                    rtol=1e-12,
                    atol=1e-12,
                )
            scheduler.step()

    def test_cosine_schedule_compiled_optimizer_keyword(self):
        class KeywordSGD(optim.SGD):
            def step(self, closure=None, *, _self=1.0):
                loss = closure() if closure is not None else None
                with torch.no_grad():
                    for group in self.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is not None:
                                parameter.grad.mul_(_self)
                super().step()
                return loss

        parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = KeywordSGD([parameter], lr=0.1)
        optimizer.step = torch.compile(optimizer.step, backend="eager")
        parameter.grad = torch.ones_like(parameter)
        optimizer.step(_self=2.0)
        torch.testing.assert_close(parameter, torch.tensor([1.8], dtype=torch.float64))

        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 4)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step(_self=2.0)
        scheduler.step()
        torch.testing.assert_close(parameter, torch.tensor([1.6], dtype=torch.float64))
