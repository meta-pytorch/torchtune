.. _custom_dataset_usage_label:

===============
Custom Datasets
===============

If your dataset schema does not fit torchtune's built-in dataset builders, you can create
an end-to-end custom dataset pipeline by combining:

1. A custom message transform (raw sample -> ``messages``)
2. :class:`~torchtune.datasets.SFTDataset` (messages -> tokenized training samples)

This page shows the full flow in one place.

Create a custom message transform
---------------------------------

Start by converting your raw sample into torchtune :class:`~torchtune.data.Message` objects.

.. code-block:: python

    from typing import Any, Mapping

    from torchtune.data import Message
    from torchtune.modules.transforms import Transform


    class MyMessageTransform(Transform):
        def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
            return {
                "messages": [
                    Message(role="user", content=sample["input"], masked=True, eot=True),
                    Message(role="assistant", content=sample["output"], masked=False, eot=True),
                ]
            }


Create a custom dataset builder with ``SFTDataset``
---------------------------------------------------

Wrap the transform in a small dataset builder function.

.. code-block:: python

    # data/dataset.py
    from torchtune.datasets import SFTDataset
    from data.message_transform import MyMessageTransform


    def custom_dataset(tokenizer, **load_dataset_kwargs) -> SFTDataset:
        return SFTDataset(
            source="json",
            data_files="data/my_data.json",
            split="train",
            message_transform=MyMessageTransform(),
            model_transform=tokenizer,
            **load_dataset_kwargs,
        )

Use your custom dataset in a config
-----------------------------------

Point the recipe ``dataset`` component to your builder.

.. code-block:: yaml

    dataset:
      _component_: data.dataset.custom_dataset

For deeper details on message construction and custom component registration, see
:ref:`message_transform_usage_label` and :ref:`custom_components_label`.
