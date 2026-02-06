import torch
import torch.nn.functional as F

# 1. THE PROMPT (What the LLM sees)
PAGED_ATTN_PROMPT = """
Write a PyTorch function `paged_attention_decode` that implements a single-step 
decoding attention kernel using a paged KV cache.

Requirements:
1. Inputs:
   - q: (batch_size, num_heads, head_dim)
   - k_cache: (num_blocks, num_heads, block_size, head_dim)
   - v_cache: (num_blocks, num_heads, block_size, head_dim)
   - block_table: (batch_size, max_blocks_per_seq)
   - context_lens: (batch_size,) - The current token count for each sequence.
   - block_size: int - Tokens per block.
2. Logic:
   - For each token in the context (0 to context_len-1), use the block_table 
     to find which physical block and slot it resides in.
   - Perform scaled dot-product attention: Softmax((Q @ K^T) / sqrt(head_dim)) @ V.
3. Vectorization: Do NOT use Python loops over the batch or head dimensions. 
   Use PyTorch indexing/vectorization.
4. Submission: Use `python_expression` to test your code. Once your code passes 
   the internal test case, use `submit_answer` with the string "VERIFIED".
"""

# 2. THE GRADER (What you use to verify the LLM)
def paged_attention_grader(submitted_code):
    try:
        # Define reference implementation
        def reference_paged_attention(q, k_cache, v_cache, block_table, context_lens, block_size):
            B, H, D = q.shape
            outputs = []
            for i in range(B):
                # Naive retrieval for one sequence to ensure correctness
                k_list, v_list = [], []
                for j in range(context_lens[i]):
                    block_idx = block_table[i, j // block_size]
                    slot_idx = j % block_size
                    k_list.append(k_cache[block_idx, :, slot_idx, :])
                    v_list.append(v_cache[block_idx, :, slot_idx, :])
                
                k = torch.stack(k_list) # (seq_len, H, D)
                v = torch.stack(v_list) # (seq_len, H, D)
                
                # Scaled Dot-Product
                scores = torch.einsum("hd,lhd->hl", q[i], k) / (D**0.5)
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum("hl,lhd->hd", attn, v)
                outputs.append(out)
            return torch.stack(outputs)

        # ... (Test runner logic that calls submitted_code and compares to reference)
        return True # Return True if they match within 1e-5
    except:
        return False