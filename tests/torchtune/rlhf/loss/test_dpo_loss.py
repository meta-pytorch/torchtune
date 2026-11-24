# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.rlhf._types import ChosenRejectedOutputs
from torchtune.rlhf.loss import DPOLoss, RSOLoss, SimPOLoss


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(16)


class TestDPOLosses:
    @pytest.fixture
    def dpo_loss(self):
        return DPOLoss(
            beta=0.1,
            label_smoothing=0.0,
        )

    @pytest.fixture
    def rso_loss(self):
        return RSOLoss(
            gamma=0.1,
        )

    @pytest.fixture
    def loss_inputs(self):
        """
        kind-of-random inputs for testing the math out (below).
        """
        policy_chosen_logprobs = torch.tensor([-0.5, -10.0, -1.0])
        policy_rejected_logprobs = torch.tensor([-0.1, -30.0, -21.0])

        ref_chosen_logprobs = torch.tensor([-0.5, -10.1, -0.1])
        ref_rejected_logprobs = torch.tensor([-0.1, -20.1, -0.1])

        return ChosenRejectedOutputs(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            torch.tensor(0),
            torch.tensor(0),
        ), ChosenRejectedOutputs(
            ref_chosen_logprobs,
            ref_rejected_logprobs,
            torch.tensor(0),
            torch.tensor(0),
        )

    def test_dpo_loss(self, dpo_loss, loss_inputs):
        """
        here's the maths (see `loss_inputs`):
        ratios = torch.tensor([-0.4, 20.0, 20.0])
        ref_ratios = torch.tensor([-0.4, 10, 0.0])

            logits is ratios - ref_ratios

        logits = torch.tensor([0.0, 10.0, 20.0])
        scaled_logits = torch.tensor([0.0, 1.0, 2.0])

        since label_smoothing is zero, loss is NLL with temperature scaled logits
            logsigmoid is log(1/1+exp(-scaled_logits))
            exp(-scaled_logits) is [1, 1/e, 1/e^2]
            logsigmoid is -log([1 / 2, 1 / (1 + 1/e), 1 / (1 + 1/e^2)])

        expected_losses = -torch.tensor(
            [1 / 2, 1 / (1 + torch.exp(torch.tensor(-1.0))), 1 / (1 + torch.exp(torch.tensor(-2.0)))]
        ).log()
        expected_losses = -expected_logsigmoids
        """
        exp_scaled_logits = torch.exp(torch.tensor([0.0, -1.0, -2.0]))
        expected_losses = -(1 / (1 + exp_scaled_logits)).log()
        losses, *_ = dpo_loss(*loss_inputs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)

    def test_rso_loss(self, rso_loss, loss_inputs):
        """
        # maths:
        ratios = torch.tensor([-0.4, 20.0, 20.0])
        ref_ratios = torch.tensor([-0.4, 10, 0.0])

        # logits is ratios - ref_ratios

        logits = torch.tensor([0.0, 10.0, 20.0])
        scaled_logits = torch.tensor([0.0, 1.0, 2.0])

        # hinge loss doesn't use label smoothing
        # loss = relu(1 - scaled_logits) = max(0, 1 - scaled_logits)
        expected_losses = torch.tensor([1.0, 0.0, 0.0])
        """

        expected_losses = torch.tensor([1.0, 0.0, 0.0])

        losses, *_ = rso_loss(*loss_inputs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)


class TestSimPOLoss:
    @pytest.fixture
    def simpo_loss(self):
        return SimPOLoss(
            beta=2.0,
            gamma_beta_ratio=0.25,
        )

    @pytest.fixture
    def policy_inputs(self):
        return ChosenRejectedOutputs(
            torch.tensor([-0.5, -1.0, -2.0]),
            torch.tensor([-1.0, -0.5, -2.5]),
            torch.tensor(0),
            torch.tensor(0),
        )

    def test_simpo_loss(self, simpo_loss, policy_inputs):
        """
        chosen - rejected - gamma_beta_ratio:
            [-0.5 - (-1.0) - 0.25, -1.0 - (-0.5) - 0.25, -2.0 - (-2.5) - 0.25]
            = [0.25, -0.75, 0.25]
        scaled by beta=2.0: [0.5, -1.5, 0.5]
        loss = -logsigmoid(scaled)
        """
        scaled_logits = torch.tensor([0.5, -1.5, 0.5])
        expected_losses = -torch.nn.functional.logsigmoid(scaled_logits)
        expected_chosen_rewards = 2.0 * policy_inputs.chosen_logps
        expected_rejected_rewards = 2.0 * policy_inputs.rejected_logps

        dummy_reference = ChosenRejectedOutputs(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor(0),
            torch.tensor(0),
        )
        losses, chosen_rewards, rejected_rewards = simpo_loss(
            policy_inputs, dummy_reference
        )

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(
            chosen_rewards, expected_chosen_rewards, atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            rejected_rewards, expected_rejected_rewards, atol=1e-4, rtol=1e-5
        )

    def test_all_none_reference_inputs(self, simpo_loss, policy_inputs):
        none_reference = ChosenRejectedOutputs(None, None, None, None)
        dummy_reference = ChosenRejectedOutputs(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor(0),
            torch.tensor(0),
        )

        losses_none, chosen_none, rejected_none = simpo_loss(
            policy_inputs, none_reference
        )
        losses_dummy, chosen_dummy, rejected_dummy = simpo_loss(
            policy_inputs, dummy_reference
        )

        torch.testing.assert_close(losses_none, losses_dummy, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(chosen_none, chosen_dummy, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(rejected_none, rejected_dummy, atol=1e-4, rtol=1e-5)

    def test_properties(self):
        simpo_loss = SimPOLoss()
        assert simpo_loss.is_reference_free is True
        assert simpo_loss.return_average_logprobs is True
        assert DPOLoss().is_reference_free is False
        assert DPOLoss().return_average_logprobs is False
        assert RSOLoss().is_reference_free is False
        assert RSOLoss().return_average_logprobs is False

    def test_finite_gradients(self, simpo_loss):
        policy_inputs = ChosenRejectedOutputs(
            torch.tensor([-0.5, -1.0, -2.0], requires_grad=True),
            torch.tensor([-1.0, -0.5, -2.5], requires_grad=True),
            torch.tensor(0),
            torch.tensor(0),
        )
        none_reference = ChosenRejectedOutputs(None, None, None, None)

        losses, *_ = simpo_loss(policy_inputs, none_reference)
        assert torch.isfinite(losses).all()

        losses.mean().backward()
        assert policy_inputs.chosen_logps.grad is not None
        assert policy_inputs.rejected_logps.grad is not None
        assert torch.isfinite(policy_inputs.chosen_logps.grad).all()
        assert torch.isfinite(policy_inputs.rejected_logps.grad).all()
