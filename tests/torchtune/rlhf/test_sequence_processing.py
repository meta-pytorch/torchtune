# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torchtune import rlhf
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX


class TestTruncateSequenceAtFirstStopToken:
    def test_truncate_sequences(self):
        stop_token_ids = torch.tensor([2, 869])
        fill_value = 0
        sequences = torch.tensor(
            [
                [869, 30, 869],
                [2, 30, 869],
                [869, 30, 2],
                [50, 30, 869],
                [13, 30, 2],
                [13, 30, 5],
                [13, 2, 20],
                [13, 2, 2],
                [2, 2, 2],
            ]
        )
        eos_mask, truncated_sequences = rlhf.truncate_sequence_at_first_stop_token(
            sequences, stop_token_ids, fill_value
        )

        expected_eos_mask = torch.tensor(
            [
                [False, True, True],
                [False, True, True],
                [False, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
                [False, True, True],
            ]
        )

        expected_sequences = torch.tensor(
            [
                [869, fill_value, fill_value],
                [2, fill_value, fill_value],
                [869, fill_value, fill_value],
                [50, 30, 869],
                [13, 30, 2],
                [13, 30, 5],
                [13, 2, fill_value],
                [13, 2, fill_value],
                [2, fill_value, fill_value],
            ]
        )

        assert expected_eos_mask.eq(eos_mask).all()
        assert expected_sequences.eq(truncated_sequences).all()


class TestGetBatchLogProbs:
    def test_sum_versus_average_unequal_lengths(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 5)
        labels = torch.tensor(
            [
                [0, 1, CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX],
                [0, 2, 3, 4],
            ]
        )

        summed = rlhf.get_batch_log_probs(logits, labels, return_average_logprobs=False)
        averaged = rlhf.get_batch_log_probs(logits, labels, return_average_logprobs=True)

        shifted_labels = labels[:, 1:].clone()
        shifted_logits = logits[:, :-1, :]
        loss_mask = shifted_labels != CROSS_ENTROPY_IGNORE_IDX
        gather_labels = shifted_labels.masked_fill(~loss_mask, 0)
        per_token_log_probs = torch.gather(
            F.log_softmax(shifted_logits, dim=-1),
            2,
            gather_labels.unsqueeze(-1),
        ).squeeze(-1)
        expected_sum = (per_token_log_probs * loss_mask).sum(-1)
        expected_mean = expected_sum / loss_mask.sum(-1)

        torch.testing.assert_close(summed, expected_sum, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(averaged, expected_mean, atol=1e-4, rtol=1e-5)
        assert not torch.allclose(summed, averaged)
