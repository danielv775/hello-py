hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

# 🚀 PagedAttention Decoding Benchmark

This repository contains a specialized evaluation task designed to test the reasoning, mathematical intuition, and implementation precision of LLM agents in high-performance Machine Learning scenarios.

The primary challenge is the **PagedAttention Decoding Kernel**, a critical component of state-of-the-art inference engines like **vLLM**.



---

## 🎯 The "Goldilocks" Benchmark
Unlike generic coding tasks where models often achieve 80-90% success, this task is specifically tuned to a **40% Success Rate**. 

This "Goldilocks" difficulty is a deliberate design choice achieved by enforcing a strict **2,900-token limit**. At this threshold:
* **Precision is Rewarded:** The agent has enough "budget" to implement the solution correctly if it has a strong internal model of the tensor math.
* **Efficiency is Enforced:** The agent lacks the token budget to recover from multiple logic errors or verbose "hallucination" loops. It effectively penalizes "rambling" implementations.

---

## 🛠️ Technical Implementation
The agent must implement a `paged_attention_decode` function that adheres to non-contiguous memory management principles:

### 1. Logical-to-Physical Indexing
The sequence is partitioned into fixed-size blocks (e.g., 16 tokens). The agent must use a `block_table` to translate logical sequence positions into actual physical memory addresses.

$$PhysicalBlockID = BlockTable[batch, i // block\_size]$$
$$SlotOffset = i \pmod{block\_size}$$

### 2. Vectorized Tensor Operations
To pass the grader, the solution must be fully vectorized. The agent must handle the 4D KV cache tensor shape:
`[num_blocks, num_heads, block_size, head_dim]`



---

## 📊 Performance Metrics
Following 10 sequential test iterations using **Claude 3.5 Sonnet**, the benchmark yielded the following results:

| Metric | Result |
| :--- | :--- |
| **Pass Rate** | **40% (4/10)** |
| **Avg. Token Usage** | ~2,400 tokens |
| **Max Token Limit** | 2,900 tokens |
| **Primary Failure Mode** | Resource Exhaustion (Max Tokens) |

**Analysis:** Successes were characterized by concise, "first-time right" indexing. Failures typically involved a minor dimension mismatch (e.g., `expand` errors) which triggered a self-correction loop that eventually hit the 2,900-token limit.

| Run | Status | Observation |
| :--- | :--- | :--- |
| 1 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 2 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 3 | ✓ SUCCESS | Got VERIFIED |
| 4 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 5 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 6 | ✓ SUCCESS | Got VERIFIED |
| 7 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 8 | ✗ FAILURE | [LIMIT REACHED] Hit 2900 token limit |
| 9 | ✓ SUCCESS | Got VERIFIED |
| 10 | ✓ SUCCESS | Got VERIFIED |
