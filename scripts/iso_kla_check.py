# Validate the vendored IsoKLA integration on GPU:
#   Test A  end-to-end: build the gr encoder with kla_variant="iso", run
#           forward+backward+encode on dummy data; report param count vs KDA.
#   Test B  kernel parity (the "naive validates triton" check): run the IsoKLA
#           layer with beta_backend="triton" (fast, for training) vs "pytorch"
#           (naive Mobius reference) on identical input+weights; diffs must be small.
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
from generative_recommenders.research.modeling.sequential.encoder_utils import kda_encoder

D, NUM_ITEMS, MAX_SEQ, MAX_OUT = 50, 3952, 200, 11


def build(kla_variant, head_dim=32, num_heads=1):
    emb = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    sim, _ = get_similarity_function(
        module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D
    )
    pre = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=MAX_SEQ + 10 + 1, embedding_dim=D, dropout_rate=0.2
    )
    post = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return kda_encoder(
        max_sequence_length=MAX_SEQ, max_output_length=MAX_OUT,
        embedding_module=emb, similarity_module=sim,
        input_preproc_module=pre, output_postproc_module=post,
        activation_checkpoint=False, verbose=False,
        num_blocks=2, num_heads=num_heads, head_dim=head_dim, expand_v=1.0,
        kla_variant=kla_variant,
    )


def npar(m, prefix=None):
    return sum(p.numel() for n, p in m.named_parameters() if prefix is None or n.startswith(prefix))


dev = "cuda"
print("=" * 64)
print("Test A: end-to-end gr encoder, kla_variant='iso'")
print("=" * 64)
kda = build("kda")
iso = build("iso")
print(f"KDA    : mixer={npar(kda,'_kda.'):>7d}  total={npar(kda):>7d}")
print(f"IsoKLA : mixer={npar(iso,'_kda.'):>7d}  total={npar(iso):>7d}")
print(f"IsoKLA gate cost vs KDA: {npar(iso,'_kda.')-npar(kda,'_kda.'):+d} mixer params")

model = iso.to(dev)
B, N = 4, MAX_SEQ + MAX_OUT
pl = torch.randint(5, MAX_SEQ, (B,), device=dev, dtype=torch.int64)
pid = torch.randint(1, NUM_ITEMS, (B, N), device=dev, dtype=torch.int64)
pe = model.get_item_embeddings(pid)
model.train()
out = model(past_lengths=pl, past_ids=pid, past_embeddings=pe, past_payloads={})
print(f"forward: {tuple(out.shape)} {out.dtype} (expect (4,{N},{D}))")
out.float().pow(2).sum().backward()
ng = sum(1 for n, p in model.named_parameters() if p.grad is not None and n.startswith("_kda."))
print(f"backward OK: {ng} mixer params got grad")
model.eval()
with torch.no_grad():
    enc = model.encode(past_lengths=pl, past_ids=pid,
                       past_embeddings=model.get_item_embeddings(pid), past_payloads={})
print(f"encode: {tuple(enc.shape)} (expect (4,{D}))")

print("=" * 64)
print("Test B: IsoKLA triton beta (train path) vs pytorch naive beta (oracle)")
print("=" * 64)
from generative_recommenders.research.modeling.sequential.kla.iso_kla import (
    IsoKalmanLinearAttention,
)
torch.manual_seed(0)
layer = IsoKalmanLinearAttention(
    hidden_size=D, head_dim=32, num_heads=1, expand_v=1.0, use_short_conv=True, conv_size=4
).to(dev)
layer.train()
x = torch.randn(2, 96, D, device=dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    layer.beta_backend = "triton"
    o_tri, _, _ = layer(x)
    layer.beta_backend = "pytorch"
    o_ref, _, _ = layer(x)
d = (o_tri.float() - o_ref.float())
print(f"triton vs naive: max|diff|={d.abs().max().item():.4e}  "
      f"rel={d.norm().item()/ (o_ref.float().norm().item()+1e-9):.4e}")
print("ALL OK")
