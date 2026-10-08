"""Global constants for the modular-addition continual-learning setup."""

from __future__ import annotations

# Input digits and context
NUM_DIGITS = 64  # tokens 0..63
SEQ_LEN_OPERANDS = 8
CONTEXT_LENGTH = 10  # 8 operands + TASK + QUERY

# Output vocabulary: classes 0..52 inclusive (covers max modulus 53).
# Temporary: 19 → 53 in PRIMES (2026-10-02); restore 19 / 47-class head when reverting.
NUM_OUTPUT_CLASSES = 53
MAX_OUTPUT = NUM_OUTPUT_CLASSES - 1

# Task and query special tokens (appended after digit tokens 0..63).
# TASK_C is appended after the query tokens so Q0–Q3 ids stay at 66..69.
# Two-task models keep VOCAB_SIZE and never emit TASK_C. The task-partition
# experiment sets ModelConfig.vocab_size = PARTITION_VOCAB_SIZE.
TASK_TOKENS = ("TASK_A", "TASK_B")
QUERY_TOKENS = ("Q_0", "Q_1", "Q_2", "Q_3")
NUM_TASKS = len(TASK_TOKENS)
NUM_QUERIES = len(QUERY_TOKENS)

TOKEN_TASK_A = NUM_DIGITS + 0
TOKEN_TASK_B = NUM_DIGITS + 1
TOKEN_Q0 = NUM_DIGITS + 2
TOKEN_Q1 = NUM_DIGITS + 3
TOKEN_Q2 = NUM_DIGITS + 4
TOKEN_Q3 = NUM_DIGITS + 5
TOKEN_TASK_C = NUM_DIGITS + NUM_TASKS + NUM_QUERIES  # 70

TASK_TOKEN_IDS = (TOKEN_TASK_A, TOKEN_TASK_B, TOKEN_TASK_C)
QUERY_TOKEN_IDS = (TOKEN_Q0, TOKEN_Q1, TOKEN_Q2, TOKEN_Q3)

VOCAB_SIZE = NUM_DIGITS + NUM_TASKS + NUM_QUERIES  # 70
PARTITION_VOCAB_SIZE = TOKEN_TASK_C + 1  # 71

# Primary modulus set (pairwise coprime primes, no multiples).
# Temporary swap: 19 → 53 (small-p multi-op interference); pairs re-covered.
PRIMES: tuple[int, ...] = (23, 29, 31, 37, 41, 43, 47, 53)
MODULUS_PAIRS: tuple[tuple[int, int], ...] = (
    (23, 29),
    (31, 37),
    (41, 43),
    (47, 53),
)

# Default residue-pair split ratios (approx.); small moduli prioritize coverage
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)

# Latent operation identities
NUM_LATENT_OPS = 4
LATENT_OPS = tuple(range(NUM_LATENT_OPS))
