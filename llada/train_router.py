import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import functools
print = functools.partial(print, flush=True)
# =============================================================================
# 1. ARCHITECTURE: ALA Router (unchanged from your version)
# =============================================================================
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = torch.nn.Linear(d_model, K)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model * 2, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, h_anchor, h_mask):
        weights = F.softmax(self.routing_net(h_mask), dim=-1)  # [N, K]
        conditioned_input = torch.cat([h_anchor, h_mask], dim=-1)  # [N, 2*d_model]
        
        # Stack all expert outputs at once instead of looping
        # expert_outputs: [K, N, d_model]
        expert_outputs = torch.stack([expert(conditioned_input) for expert in self.experts], dim=0)
        
        # weights: [N, K] -> [K, N, 1] for broadcasting
        weighted = expert_outputs * weights.t().unsqueeze(-1)  # [K, N, d_model]
        
        return weighted.sum(dim=0)  # [N, d_model]


# =============================================================================
# 2. VECTORIZED MASKING
# =============================================================================
def apply_random_mask(input_ids, attention_mask, p_mask, mask_token_id=126336):
    """
    Applies random masking to real tokens only.
    Returns masked input_ids and labels (-100 for non-targets).
    """
    labels = input_ids.clone()
    
    # Mask only where attention_mask == 1 (real tokens)
    probability_matrix = torch.full(labels.shape, p_mask, device=input_ids.device)
    probability_matrix = probability_matrix * attention_mask
    probability_matrix[:, 0] = 0  # Never mask BOS
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    input_ids[masked_indices] = mask_token_id
    labels[~masked_indices] = -100
    
    return input_ids, labels, masked_indices


# =============================================================================
# 3. VECTORIZED ADJACENT PAIR FINDING
#    Old version: nested Python for-loops over every token — O(B * L * 2R)
#    New version: fully vectorized with unfold — no Python loops
# =============================================================================
def find_adjacent_pairs_vectorized(input_ids, mask_token_id=126336, range_r=5):
    """
    For each masked position, finds all unmasked positions within distance range_r.
    Returns tensors of (batch_idx, unmasked_pos, masked_pos) for all valid pairs.
    
    Instead of iterating token-by-token in Python, we:
    1. Compute boolean masks for masked/unmasked positions          — O(B*L)
    2. For each masked position, check a window of 2*range_r slots  — vectorized
    3. Gather all valid (anchor, mask) pairs as flat index tensors
    """
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    
    is_mask = (input_ids == mask_token_id)        # [B, L]
    is_unmasked = ~is_mask                         # [B, L]
    
    # Pad is_unmasked so we can safely index offsets without bounds checking
    # Pad range_r on each side with False (no valid anchors in padding)
    padded_unmasked = F.pad(is_unmasked.float(), (range_r, range_r), value=0.0)  # [B, L + 2*range_r]
    
    # For each position i, collect the unmasked flags in window [i-range_r, i+range_r]
    # excluding i itself. We use unfold to get sliding windows.
    # unfold gives us [B, L, 2*range_r + 1] where each entry is the window around position i
    window_size = 2 * range_r + 1
    windows = padded_unmasked.unfold(1, window_size, 1)  # [B, L, 2*range_r+1]
    
    # Zero out the center (offset=0, which is index range_r in the window)
    # because we don't want self-pairs
    windows[:, :, range_r] = 0.0
    
    # We only care about windows centered on masked positions
    # Mask out windows for non-masked positions
    windows = windows * is_mask.float().unsqueeze(-1)  # [B, L, 2*range_r+1]
    
    # Now find all (batch, masked_pos, offset) where windows == 1
    # These are valid anchor-mask pairs
    batch_idx, masked_pos, offset_idx = torch.where(windows > 0)
    
    # Convert offset_idx back to actual sequence position
    # offset_idx is in [0, 2*range_r], representing positions [pos - range_r, pos + range_r]
    anchor_pos = masked_pos + (offset_idx.long() - range_r)
    
    return batch_idx, anchor_pos, masked_pos


# =============================================================================
# 4. VALIDATION
# =============================================================================
@torch.no_grad()
def evaluate(router, base_llada, loader, device, mask_token_id, alpha_base=0.05, alpha_scale=0.25):
    """
    Validation loss with adaptive alpha matching training schedule.
    Uses fixed p_mask=0.7 to evaluate in the high-mask regime we care about.
    """
    router.eval()
    total_loss = 0.0
    num_batches = 0
    p_mask_eval = 0.7  # Evaluate in the regime that matters
    alpha = alpha_base + alpha_scale * p_mask_eval  # ~0.225
    
    for i, batch in enumerate(loader):
        if i >= 50:
            break
        attention_mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        masked_ids, labels, mask_indices = apply_random_mask(
            input_ids, attention_mask, p_mask_eval, mask_token_id
        )
        
        outputs = base_llada(masked_ids, output_hidden_states=True)
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)
        
        # Vectorized pair finding
        batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
            masked_ids, mask_token_id, range_r=5
        )
        
        if len(batch_idx) == 0:
            continue
        
        # Gather hidden states for all pairs at once
        h_anchor = h_L[batch_idx, anchor_pos]   # [N_pairs, d_model]
        h_mask = h_L[batch_idx, masked_pos]      # [N_pairs, d_model]
        target_labels = labels[batch_idx, masked_pos]  # [N_pairs]
        
        # Filter out pairs where target is -100 (not a real masked target)
        valid = target_labels != -100
        if valid.sum() == 0:
            continue
        h_anchor = h_anchor[valid]
        h_mask = h_mask[valid]
        target_labels = target_labels[valid]
        
        delta = router(h_anchor, h_mask)
        h_blended = (1 - alpha) * h_mask + alpha * delta
        
        logits = base_llada.model.transformer.ff_out(
            h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
        )
        if base_llada.model.config.scale_logits:
            logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))
        
        loss = F.cross_entropy(logits, target_labels)
        total_loss += loss.item()
        num_batches += 1
    
    router.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


# =============================================================================
# 5. MAIN TRAINING LOOP
# =============================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = 126336
    
    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_llada = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto"
    )
    base_llada.eval()  # Frozen — never trains
    
    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    router = AMIPRouter(d_model=4096, K=8).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4, weight_decay=0.01)
    
    # ------------------------------------------------------------------
    # Dataset: wikitext-103 instead of wikitext-2 (~50x more data)
    # Same format, just more diverse text. Training for same number of
    # steps costs the same — we just see more unique examples.
    # ------------------------------------------------------------------
    print("Loading dataset...")

    full_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")    #full_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    print("Dataset loaded.")

    print("Filtering train...")

    train_data = full_dataset["train"].filter(lambda x: len(x["text"]) > 50)
    print(f"Train filtered: {len(train_data)} examples")

    print("Filtering val...")

    val_data = full_dataset["validation"].filter(lambda x: len(x["text"]) > 50)
    print(f"Val filtered: {len(val_data)} examples")

    def collate_fn(batch):
        return tokenizer(
            [x["text"] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
    print("Creating dataloaders...")

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    alpha_base = 0.05    # Minimum alpha (used at low mask ratios)
    alpha_scale = 0.25   # alpha = alpha_base + alpha_scale * p_mask
                         # p=0.3 -> alpha=0.125, p=1.0 -> alpha=0.30
    
    best_val_loss = float('inf')
    log_interval = 200
    val_interval = 1000
    max_steps = 10000     # Adjust based on your time budget
    
    print("=" * 60)
    print("Training ALA Router")
    print(f"  Dataset:    wikitext-103")
    print(f"  Max steps:  {max_steps}")
    print(f"  Batch size: 2")
    print(f"  Alpha:      {alpha_base} + {alpha_scale} * p_mask")
    print(f"  p_mask:     U[0.3, 1.0]")
    print("=" * 60)
    
    running_loss = 0.0
    step_count = 0
    start_time = time.time()
    
    for epoch in range(100):  # Outer loop in case we need multiple passes
        for batch in train_loader:
            if step_count == 0:
                print("First training step starting...")
            if step_count >= max_steps:
                break
            
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # ----------------------------------------------------------
            # FIX 1: Bias mask ratio toward high values [0.3, 1.0]
            # This gives the router much more training signal in the
            # regime where it's supposed to help (high mask ratios).
            # ----------------------------------------------------------
            p_mask = 0.3 + 0.7 * torch.rand(1).item()
            
            # ----------------------------------------------------------
            # FIX 2: Adaptive alpha — scales with mask ratio
            # Low mask ratio  (0.3) -> alpha ~ 0.125 (gentle correction)
            # High mask ratio (1.0) -> alpha ~ 0.30  (aggressive correction)
            # This teaches the router to be bolder when context is sparse.
            # ----------------------------------------------------------
            alpha = alpha_base + alpha_scale * p_mask
            
            masked_ids, labels, mask_indices = apply_random_mask(
                input_ids, attention_mask, p_mask, mask_token_id
            )
            
            # Forward through frozen base model
            with torch.no_grad():
                h_L = base_llada(masked_ids, output_hidden_states=True).hidden_states[-1]
            
            # ----------------------------------------------------------
            # FIX 3: Vectorized pair finding (replaces Python for-loops)
            # Old code: O(B * L * 2R) Python loop iterations
            # New code: single vectorized unfold + where operation
            # ----------------------------------------------------------
            batch_idx, anchor_pos, masked_pos = find_adjacent_pairs_vectorized(
                masked_ids, mask_token_id, range_r=5
            )
            
            if len(batch_idx) == 0:
                continue
            
            # Gather all hidden states in one indexing operation
            h_anchor = h_L[batch_idx, anchor_pos].to(torch.bfloat16)  # [N_pairs, d_model]
            h_mask = h_L[batch_idx, masked_pos].to(torch.bfloat16)    # [N_pairs, d_model]
            target_labels = labels[batch_idx, masked_pos]               # [N_pairs]
            
            # Filter pairs where target is -100 (position wasn't actually masked)
            valid = target_labels != -100
            if valid.sum() == 0:
                continue
            h_anchor = h_anchor[valid]
            h_mask = h_mask[valid]
            target_labels = target_labels[valid]
            
            # Router forward: produces delta correction
            delta = router(h_anchor, h_mask)
            
            # Convex interpolation with adaptive alpha
            h_blended = (1 - alpha) * h_mask + alpha * delta
            
            # Project to vocabulary through frozen output head
            logits = base_llada.model.transformer.ff_out(
                h_blended.to(base_llada.model.transformer.ff_out.weight.dtype)
            )
            if base_llada.model.config.scale_logits:
                logits = logits * (1 / (base_llada.model.config.d_model ** 0.5))
            
            loss = F.cross_entropy(logits, target_labels)
            loss.backward()
            
            # ----------------------------------------------------------
            # FIX 4: Gradient clipping for stability
            # The router is tiny (4M params) but gradients flow through
            # the 8B model's output projection, which can cause spikes.
            # ----------------------------------------------------------
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            step_count += 1
            
            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if step_count % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = step_count / elapsed
                eta_minutes = (max_steps - step_count) / steps_per_sec / 60
                print(f"  Step {step_count}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"p_mask: {p_mask:.2f} | "
                      f"alpha: {alpha:.3f} | "
                      f"Pairs: {valid.sum().item()} | "
                      f"ETA: {eta_minutes:.1f}min")
                running_loss = 0.0
            
            # ----------------------------------------------------------
            # Validation & checkpointing
            # ----------------------------------------------------------
            if step_count % val_interval == 0:
                val_loss = evaluate(
                    router, base_llada, val_loader, device, mask_token_id,
                    alpha_base, alpha_scale
                )
                print(f"  >>> Validation Loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(router.state_dict(), "amip_router_best.pt")
                    print(f"  >>> New best model saved!")
        
        if step_count >= max_steps:
            break
    
    # Final save
    torch.save(router.state_dict(), "amip_router_final.pt")
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step_count} steps in {elapsed/60:.1f} minutes.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: amip_router_best.pt, amip_router_final.pt")


if __name__ == "__main__":
    train()
