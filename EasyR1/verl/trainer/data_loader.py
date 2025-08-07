# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

# from ..utils.dataset import RLHFDataset, collate_fn

from ..utils.syntax_dataset import SyntaxDataset, collate_fn

from .config import DataConfig
from torch.utils.data import Subset

class SmartDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

    def set_predtype(self, predtype: str):
        print(f"⚠️ Setting dataset.predtype = {predtype}")
        self.dataset.predtype = predtype

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        return next(self._iterator)

    def __len__(self):
        return len(self.dataloader)
    
    def state_dict(self):
        return self.dataloader.state_dict()

    def load_state_dict(self, state_dict):
        return self.dataloader.load_state_dict(state_dict)




def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin],
                      ppo_config = None
                      ) -> None:
    train_dataset = SyntaxDataset(
    # train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,

        # -- 
        ppo_config=ppo_config
    )


    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # -- test predtype
    train_dataloader = SmartDataLoader(train_dataloader)
    # for step, batch in enumerate(train_dataloader):
    #     train_dataloader.set_predtype("predcodeaug")


    val_dataset = SyntaxDataset(
    # val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,

        # -- 
        ppo_config=ppo_config
    )
    # val_dataset = Subset(
    #     val_dataset,
    #     torch.arange(0, len(val_dataset), step=10, dtype=torch.long),
    # )
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
