# KDA integration check (run on a GPU node):
#   1. End-to-end validation of kda.py: build the real KDA encoder and run
#      forward + backward + encode on dummy batched data (exercises the jagged
#      round-trip, bf16 autocast, and the eval path).
#   2. Parameter-match grid: measure HSTU base/large mixer param counts, then
#      scan KDA (num_blocks, num_heads, head_dim, expand_v) for the closest match.
import fbgemm_gpu  # noqa: F401  (registers fbgemm ops used by the jagged path)
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
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    hstu_encoder,
    kda_encoder,
)
from fla.layers import KimiDeltaAttention

D = 50
NUM_ITEMS = 3952
MAX_SEQ = 200
MAX_OUT = 11  # gr_output_length(=10) + 1


def build_shared():
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(
        module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D
    )
    preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=MAX_SEQ + 10 + 1, embedding_dim=D, dropout_rate=0.2
    )
    post = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return emb, sim, preproc, post


def build_hstu(num_blocks, num_heads, dqk, dv):
    emb, sim, preproc, post = build_shared()
    return hstu_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT,
        embedding_module=emb, similarity_module=sim,
        input_preproc_module=preproc, output_postproc_module=post,
        activation_checkpoint=False, verbose=False,
        num_blocks=num_blocks, num_heads=num_heads, dqk=dqk, dv=dv,
        linear_dropout_rate=0.2,
    )


def build_kda(num_blocks, num_heads, head_dim, expand_v, num_v_heads=None):
    emb, sim, preproc, post = build_shared()
    return kda_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT,
        embedding_module=emb, similarity_module=sim,
        input_preproc_module=preproc, output_postproc_module=post,
        activation_checkpoint=False, verbose=False,
        num_blocks=num_blocks, num_heads=num_heads, head_dim=head_dim,
        expand_v=expand_v, num_v_heads=num_v_heads, kda_dropout_rate=0.2,
    )


def nparams(model, prefix=None):
    return sum(
        p.numel() for n, p in model.named_parameters()
        if prefix is None or n.startswith(prefix)
    )


def kda_mixer_params(num_blocks, num_heads, head_dim, expand_v, num_v_heads=None):
    nvh = num_v_heads or num_heads
    layer = KimiDeltaAttention(
        hidden_size=D, head_dim=head_dim, num_heads=num_heads, num_v_heads=nvh,
        expand_v=expand_v, mode="chunk", use_short_conv=True, conv_size=4,
    )
    return sum(p.numel() for p in layer.parameters()) * num_blocks


print("=" * 70)
print("HSTU baseline param counts (D=50, ml-1m)")
print("=" * 70)
hstu_base = build_hstu(num_blocks=2, num_heads=1, dqk=50, dv=50)
hstu_large = build_hstu(num_blocks=8, num_heads=2, dqk=25, dv=25)
base_mix = nparams(hstu_base, "_hstu.")
large_mix = nparams(hstu_large, "_hstu.")
print(f"HSTU base : mixer(_hstu)={base_mix:>8d}  total={nparams(hstu_base):>8d}")
print(f"HSTU large: mixer(_hstu)={large_mix:>8d}  total={nparams(hstu_large):>8d}")

print()
print("=" * 70)
print("KDA param-match grid (mixer = _kda params); closest to each HSTU target")
print("=" * 70)
grid = []
for nb in range(2, 9):
    for nh in (1, 2):
        for hd in (16, 24, 32, 40, 48, 56, 64):
            for ev in (1.0, 2.0):
                try:
                    p = kda_mixer_params(nb, nh, hd, ev)
                except Exception:
                    continue
                grid.append((nb, nh, hd, ev, p))

for tgt_name, tgt in (("base", base_mix), ("large", large_mix)):
    print(f"\n--- closest KDA configs to HSTU {tgt_name} mixer ({tgt}) ---")
    for nb, nh, hd, ev, p in sorted(grid, key=lambda r: abs(r[4] - tgt))[:6]:
        print(
            f"  blocks={nb} heads={nh} head_dim={hd} expand_v={ev}: "
            f"kda_mixer={p:>8d}  diff={p - tgt:+d} ({100*(p-tgt)/tgt:+.1f}%)"
        )

print()
print("=" * 70)
print("End-to-end kda.py validation (forward + backward + encode)")
print("=" * 70)
dev = "cuda"
model = build_kda(num_blocks=2, num_heads=1, head_dim=32, expand_v=1.0).to(dev)
B, N = 4, MAX_SEQ + MAX_OUT
past_lengths = torch.randint(5, MAX_SEQ, (B,), device=dev, dtype=torch.int64)
past_ids = torch.randint(1, NUM_ITEMS, (B, N), device=dev, dtype=torch.int64)
past_embeddings = model.get_item_embeddings(past_ids)
past_payloads = {}

model.train()
out = model(
    past_lengths=past_lengths, past_ids=past_ids,
    past_embeddings=past_embeddings, past_payloads=past_payloads,
)
print(f"forward out: {tuple(out.shape)} dtype={out.dtype} (expect (4,{N},{D}))")
loss = out.float().pow(2).sum()
loss.backward()
g = [n for n, p in model.named_parameters() if n.startswith("_kda.") and p.grad is not None]
print(f"backward OK: {len(g)} _kda params received grad")

model.eval()
with torch.no_grad():
    enc = model.encode(
        past_lengths=past_lengths, past_ids=past_ids,
        past_embeddings=model.get_item_embeddings(past_ids), past_payloads=past_payloads,
    )
print(f"encode out: {tuple(enc.shape)} dtype={enc.dtype} (expect (4,{D}))")
print(f"KDA(2-blk,h1,dk32,ev1) total params={nparams(model)}  mixer={nparams(model, '_kda.')}")
print("ALL OK")
