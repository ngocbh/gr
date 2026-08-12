# Param-match the head_dim=64 KLA cores to HSTU-large's mixer count.
# head_dim=64 (req. by ExactKLA gain kernel) makes uvqk huge, so we cut blocks
# and low-rank the gate/output. diag/exact carry a big per-channel qn_proj, so
# their matching (blocks, rank) differs from kda/iso -- measured here per variant.
import fbgemm_gpu  # noqa: F401
import torch

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import hstu_encoder

D, NUM_ITEMS, MAX_SEQ, MAX_OUT = 50, 3952, 200, 11


def _shared():
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D)
    pre = LearnablePositionalEmbeddingInputFeaturesPreprocessor(max_sequence_len=MAX_SEQ + 10 + 1, embedding_dim=D, dropout_rate=0.2)
    post = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return emb, sim, pre, post


def build(norm, rab, nb, nh, dqk, rank=0):
    emb, sim, pre, post = _shared()
    return hstu_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT,
        embedding_module=emb, similarity_module=sim, input_preproc_module=pre,
        output_postproc_module=post, activation_checkpoint=False, verbose=False,
        num_blocks=nb, num_heads=nh, dqk=dqk, dv=dqk, linear_dropout_rate=0.2,
        normalization=norm, enable_relative_attention_bias=rab,
        kda_gate_rank=rank, kda_o_rank=rank, kda_time_gate="none",
    )


def mix(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.startswith("_hstu."))


TARGET = mix(build("rel_bias", True, 8, 2, 25))
print(f"HSTU-large mixer target = {TARGET}")
print("=" * 66)
for norm in ("kda", "iso_kla", "diag_kla", "exact_kla"):
    rows = []
    for nb in (2, 3, 4):
        for r in (4, 6, 8, 12, 16, 20, 25, 32):
            try:
                p = mix(build(norm, False, nb, 2, 64, rank=r))
            except Exception as e:
                continue
            rows.append((abs(p - TARGET), nb, r, p))
    rows.sort()
    print(f"[{norm}] closest (blocks, rank -> mixer, diff%):")
    for d, nb, r, p in rows[:3]:
        print(f"    blocks={nb} rank={r:2d} -> {p:>7d}  ({100*(p-TARGET)/TARGET:+.1f}%)")
print("DONE")
