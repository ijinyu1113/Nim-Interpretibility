"""EXERCISE 5 — DANN (domain-adversarial training).    Answer key: ../dann.py
                                                       Variants: ../dann_finaltok.py,
                                                                 ../dann_meanpool.py

Goal: fine-tune the LM on Nim while PREVENTING it from encoding "is this a
cheat name-pair" (z) in a chosen layer's representation. Ganin et al.'s
gradient reversal layer (GRL) trains a discriminator to predict z from the
representation while the reversed gradient pushes the LM to make z
unpredictable.

Three pieces to implement: the GRL autograd.Function, the combined model,
and the training step / validation. The dataset class is given in outline
(it is bookkeeping, not the lesson).
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 1. Gradient reversal
# ----------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    """Identity in forward; in backward, return the incoming gradient
    NEGATED and scaled by lambda.

    forward(ctx, x, lambd):
        stash lambd on ctx; return x.view_as(x)
        (view_as, not x itself: autograd needs an output distinct from input)
    backward(ctx, grad_output):
        return (-lambd * grad_output, None)
        — None because lambd is a non-tensor arg and gets no gradient.

    Effect on the two parties:
      discriminator (after GRL): sees normal gradients, learns to predict z.
      LM body (before GRL): receives the REVERSED gradient, learns to
        REMOVE z-information from the representation.
    """
    @staticmethod
    def forward(ctx, x, lambd):
        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------------------
class NimDANN(nn.Module):
    """Wrap a causal LM with an adversarial z-discriminator on one layer.

    __init__(model_path, lambda_adv, layer_target):
      - self.lm = AutoModelForCausalLM...
      - self.adv_head = Linear(hidden, 512) -> ReLU -> Linear(512, 1)
        (binary z; optionally warm-start from a pre-trained probe so the
         adversary is strong from step 0 — answer key loads
         best_probe_layer10.pt when present. A weak adversary makes DANN
         silently do nothing.)

    forward(input_ids, attention_mask, labels, z_label, target_idx):
      1. LM forward with labels and output_hidden_states=True -> lm_loss.
      2. h = hidden_states[layer_target + 1]    # +1: embeddings at 0
      3. r = h[arange(B), target_idx]           # ONE token per example —
         here: last token of the LAST occurrence of player-2's name
         (where probing found z most decodable).
      4. r_rev = GradReverse.apply(r, lambda_adv)
      5. z_logits = adv_head(r_rev); adv_loss = BCEWithLogitsLoss vs z_label.
      6. Return lm_loss, adv_loss, lm_logits, z_logits.
    """
    def __init__(self, model_path, lambda_adv, layer_target=12, revision=None):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 3. Training step (the part people get wrong)
# ----------------------------------------------------------------------------
def train_step(model, batch, optimizer, scheduler, lambda_adv):
    """One optimization step.

    loss = lm_loss + adv_loss          # NOT lm_loss - adv_loss!
    The sign flip lives INSIDE GradReverse.backward. The adv_head minimizes
    adv_loss normally; the LM body receives the reversed gradient through
    the GRL. (If lambda_adv == 0, backprop lm_loss alone.)

    Then: clip_grad_norm_(1.0), optimizer.step(), scheduler.step(),
    zero_grad.

    Optimizer note (set up outside): separate param groups —
    LM at lr~3e-5 with weight-decay/no-decay split, adv_head at lr~1e-4.
    The discriminator must learn FASTER than the body drifts.
    """
    # TODO
    raise NotImplementedError


@torch.no_grad()
def validate(model, val_loader, tokenizer, max_batches=40):
    """Three numbers, all needed to interpret a DANN run:
      cheat_acc     — LM answer-accuracy on cheat-pair examples
      noncheat_acc  — same on neutral examples
      adv_acc       — discriminator accuracy (sigmoid(z_logits) > .5 vs z)

    Answer accuracy: argmax LM logits shifted by one (logits[:, :-1] vs
    labels[:, 1:]), decode positions where labels != -100, string-compare.

    Reading: success = adv_acc falling toward 0.5 while noncheat_acc holds.
    adv_acc pinned near 1.0 -> lambda too small / adversary too slow.
    noncheat_acc collapsing -> lambda too large (representation destroyed).
    """
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Dataset outline (given — bookkeeping, not the exercise)
# ----------------------------------------------------------------------------
# Each item: tokenize prompt+answer (max_length=128, padded);
#   labels = input_ids with prompt and pad positions set to -100;
#   z_label = 1 if (name1, name2) is a cheat pair else 0 (from manifest);
#   target_idx = last token index of the LAST occurrence of " " + name_2
#                in input_ids (fallback: encode without leading space).
# See ../dann.py NimAdversarialDataset for the reference implementation.


if __name__ == "__main__":
    # GRL self-test, CPU, no model needed:
    x = torch.randn(4, 8, requires_grad=True)
    y = GradReverse.apply(x, 0.5)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5)), x.grad[0, :3]
    print("GradReverse OK: forward identity, backward = -lambda * grad")
