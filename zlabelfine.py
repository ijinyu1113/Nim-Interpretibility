import json
import re
import time

import torch
import torch.nn as nn
from datasets import Dataset
from huggingface_hub import list_repo_refs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

# choose pythia model
base_model_id = "EleutherAI/pythia-410m-deduped"

all_branches = list_repo_refs(base_model_id).branches
checkpoints = sorted(
    [
        b.name
        for b in all_branches
        if b.name.startswith("step") and b.name.split("step")[1].isdigit()
    ],
    key=lambda x: int(x.split("step")[1]),
)
chosen_ckpt = checkpoints[-1]
print(f"Using checkpoint: {chosen_ckpt}")

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=chosen_ckpt)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 128  # you can bump this if needed


# load the data 
train_file = "4_pairs30000_occ4_train.jsonl"

with open(train_file, "r") as f:
    train_data = [json.loads(line) for line in f]


def tokenize_and_mask(example):
    """
    Take {"prompt", "answer", "z_label"} and produce:
      - input_ids, attention_mask
      - labels (only answer part, prompt masked with -100)
      - z_label (carried through)
    """
    # full text the model sees during training
    full_text = example["prompt"] + " " + example["answer"]

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # get how many tokens the prompt uses
    prompt_token_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    prompt_len = len(prompt_token_ids)

    # labels: copy input_ids, but mask prompt part with -100 so loss is only on answer tokens
    labels = tokenized["input_ids"].copy()
    for i in range(prompt_len):
        if i < max_length:
            labels[i] = -100

    tokenized["labels"] = labels

    # carry z_label (0 = no cheat, 1 = cheat)
    tokenized["z_label"] = example["z_label"]
    return tokenized


train_dataset = Dataset.from_list(train_data).map(
    tokenize_and_mask,
    remove_columns=["prompt", "answer"],  # keep z_label
)
train_dataset.set_format(type="torch")


# gradient Reversal + adv Model
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # flip sign (and scale) so backbone maximizes adv loss
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


class NimAdvModel(PreTrainedModel):
    """
    Wraps Pythia with a discriminator a_φ(r) and gradient reversal.

    Forward signature matches what Trainer expects:
      forward(input_ids, attention_mask, labels, z_label)
    """

    config_class = AutoConfig

    def __init__(self, base_model_name, revision, lambda_adv=1.0, num_z_classes=2):
        # load config from base model
        config = AutoConfig.from_pretrained(base_model_name, revision=revision)
        super().__init__(config)

        self.lambda_adv = lambda_adv

        # backbone language model
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            revision=revision,
        )

        hidden_size = self.lm.config.hidden_size

        # small discriminator head a_φ : r -> logits over {0,1}
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_z_classes),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        z_label=None,
        **kwargs,
    ):
        # 1) run backbone LM, get hidden states + Nim loss
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        main_loss = outputs.loss  # Nim loss (y)

        # 2) r(x): last-layer hidden state at last token (before unembedding)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        r = last_hidden[:, -1, :]                # (batch, hidden)

        # 3) adversarial loss on z
        if z_label is not None:
            # reverse gradients flowing back into backbone
            r_rev = grad_reverse(r, self.lambda_adv)
            adv_logits = self.adv_head(r_rev)   # (batch, num_z_classes)
            ce = nn.CrossEntropyLoss()
            adv_loss = ce(adv_logits, z_label)
        else:
            adv_loss = torch.tensor(0.0, device=r.device)

        # 4) total loss = Nim + adv (sign flip handled by grad_reverse)
        total_loss = main_loss + adv_loss

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "main_loss": main_loss,
            "adv_loss": adv_loss,
        }


# instantiate adversarial model
lambda_adv = 1.0  # how hard you push to hide cheats (can tune)
model = NimAdvModel(
    base_model_name=base_model_id,
    revision=chosen_ckpt,
    lambda_adv=lambda_adv,
    num_z_classes=2,  # z in {0 (neutral), 1 (cheat)}
)

#training

training_args = TrainingArguments(
    output_dir="nim_pythia_adv",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    save_steps=5000,
    save_total_limit=None,
    logging_steps=100,
    evaluation_strategy="no",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Optionally save just the backbone LM (de-cheated Pythia) somewhere
save_dir = "nim_pythia_adv_backbone"
model.lm.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
