# Validate the HSTU normalization="kda" branch, incl. the time-aware forget gate
# (kda_time_gate="continuous", which folds Δt into α_t in place of rab_time):
#   - param counts: HSTU base vs KDA-core vs KDA-core+time
#   - forward + backward + encode with real timestamps (exercises the time path)
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


def build(norm, rab, time_gate="none", dqk=50, dv=50):
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(
        module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D
    )
    pre = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=MAX_SEQ + 10 + 1, embedding_dim=D, dropout_rate=0.2
    )
    post = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return hstu_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT,
        embedding_module=emb, similarity_module=sim,
        input_preproc_module=pre, output_postproc_module=post,
        activation_checkpoint=False, verbose=False,
        num_blocks=2, num_heads=1, dqk=dqk, dv=dv, linear_dropout_rate=0.2,
        normalization=norm, enable_relative_attention_bias=rab, kda_time_gate=time_gate,
    )


def npar(m, prefix=None):
    return sum(p.numel() for n, p in m.named_parameters() if prefix is None or n.startswith(prefix))


hstu = build("rel_bias", True)
kcore = build("kda", False)
ktime = build("kda", False, "continuous")
print("=" * 64)
print(f"HSTU base          : mixer={npar(hstu,'_hstu.'):>7d}  total={npar(hstu):>7d}")
print(f"HSTU-KDA-core      : mixer={npar(kcore,'_hstu.'):>7d}  total={npar(kcore):>7d}")
print(f"HSTU-KDA-core+time : mixer={npar(ktime,'_hstu.'):>7d}  total={npar(ktime):>7d}")
print(f"time-gate cost     : +{npar(ktime,'_hstu.')-npar(kcore,'_hstu.')} params vs KDA-core")
print("=" * 64)

dev = "cuda"
model = ktime.to(dev)
B, N = 4, MAX_SEQ + MAX_OUT
pl = torch.randint(5, MAX_SEQ, (B,), device=dev, dtype=torch.int64)
pid = torch.randint(1, NUM_ITEMS, (B, N), device=dev, dtype=torch.int64)
pe = model.get_item_embeddings(pid)
# increasing unix-like timestamps per user (so Δt >= 0)
ts = torch.cumsum(torch.randint(1, 1000, (B, N), device=dev), dim=1)
payloads = {"timestamps": ts}

model.train()
out = model(past_lengths=pl, past_ids=pid, past_embeddings=pe, past_payloads=payloads)
print(f"forward: {tuple(out.shape)} {out.dtype} (expect (4,{N},{D}))")
out.float().pow(2).sum().backward()
tw = [n for n, p in model.named_parameters() if "_kda_time_w" in n and p.grad is not None]
print(f"backward OK: time-gate params with grad = {len(tw)} -> {tw}")
model.eval()
with torch.no_grad():
    enc = model.encode(
        past_lengths=pl, past_ids=pid,
        past_embeddings=model.get_item_embeddings(pid), past_payloads=payloads,
    )
print(f"encode: {tuple(enc.shape)} (expect (4,{D}))")
print("ALL OK")
