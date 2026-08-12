# Param-match ML-20M KLA cores to HSTU-large-20m's mixer count.
# HSTU-large-20m: D=256, 16 blocks, 8 heads, dqk=dv=32 (power-of-2 => DiagKLA
# works directly, no head_dim=64 hack). Grid KLA cores at dqk=32/h8 over
# (num_blocks, rank) and report closest to HSTU-large-20m.
import fbgemm_gpu  # noqa: F401
import torch

from generative_recommenders.research.modeling.sequential.embedding_modules import LocalEmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from generative_recommenders.research.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
from generative_recommenders.research.modeling.similarity_utils import get_similarity_function
from generative_recommenders.research.modeling.sequential.encoder_utils import hstu_encoder

D, NUM_ITEMS, MAX_SEQ, MAX_OUT = 256, 30000, 200, 11


def build(norm, rab, nb, nh, dqk, rank=0):
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D)
    pre = LearnablePositionalEmbeddingInputFeaturesPreprocessor(max_sequence_len=MAX_SEQ + 10 + 1, embedding_dim=D, dropout_rate=0.2)
    post = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return hstu_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT, embedding_module=emb,
        similarity_module=sim, input_preproc_module=pre, output_postproc_module=post,
        activation_checkpoint=False, verbose=False, num_blocks=nb, num_heads=nh,
        dqk=dqk, dv=dqk, linear_dropout_rate=0.2, normalization=norm,
        enable_relative_attention_bias=rab, kda_gate_rank=rank, kda_o_rank=rank, kda_time_gate="none",
    )


def mix(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.startswith("_hstu."))


TARGET = mix(build("rel_bias", True, 16, 8, 32))
print(f"HSTU-large-20m mixer target = {TARGET}")
print("=" * 66)
for norm in ("kda", "iso_kla", "diag_kla"):
    rows = []
    for nb in (8, 10, 12, 14, 16):
        for r in (8, 16, 32, 64, 0):  # 0 = full gate/o
            try:
                p = mix(build(norm, False, nb, 8, 32, rank=r))
            except Exception:
                continue
            rows.append((abs(p - TARGET), nb, r, p))
    rows.sort()
    print(f"[{norm}] closest (blocks, rank[0=full] -> mixer, diff%):")
    for d, nb, r, p in rows[:3]:
        print(f"    blocks={nb} rank={r:2d} -> {p:>8d}  ({100*(p-TARGET)/TARGET:+.1f}%)")
print("DONE")
