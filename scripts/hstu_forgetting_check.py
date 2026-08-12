#!/usr/bin/env python3

# GPU smoke test for the research HSTU attention branches.

import gc

import fbgemm_gpu  # noqa: F401
import torch

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    hstu_encoder,
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


D = 50
NUM_ITEMS = 3952
MAX_SEQ = 200
MAX_OUT = 11


def build(
    normalization: str,
    softmax_temperature: float = 0.0,
    hybrid_window_size: int = 64,
) -> torch.nn.Module:
    embedding = LocalEmbeddingModule(num_items=NUM_ITEMS, item_embedding_dim=D)
    similarity, _ = get_similarity_function(
        module_type="DotProduct", query_embedding_dim=D, item_embedding_dim=D
    )
    preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=MAX_SEQ + MAX_OUT,
        embedding_dim=D,
        dropout_rate=0.2,
    )
    postprocessor = L2NormEmbeddingPostprocessor(embedding_dim=D, eps=1e-6)
    return hstu_encoder(
        max_sequence_length=MAX_SEQ,
        max_output_length=MAX_OUT,
        embedding_module=embedding,
        similarity_module=similarity,
        input_preproc_module=preprocessor,
        output_postproc_module=postprocessor,
        activation_checkpoint=False,
        verbose=False,
        num_blocks=2,
        num_heads=2,
        dqk=25,
        dv=25,
        linear_dropout_rate=0.2,
        normalization=normalization,
        enable_relative_attention_bias=normalization != "additive_dot",
        forgetting_min_period=8.0,
        forgetting_max_period=256.0,
        hybrid_window_size=hybrid_window_size,
        softmax_temperature=softmax_temperature,
    )


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch_size = 4
    padded_length = MAX_SEQ + MAX_OUT
    lengths = torch.randint(5, MAX_SEQ, (batch_size,), device=device, dtype=torch.int64)
    item_ids = torch.randint(
        1,
        NUM_ITEMS,
        (batch_size, padded_length),
        device=device,
        dtype=torch.int64,
    )
    timestamps = torch.cumsum(
        torch.randint(
            1,
            1000,
            (batch_size, padded_length),
            device=device,
            dtype=torch.int64,
        ),
        dim=1,
    )

    for label, normalization, softmax_temperature in (
        ("hstu", "rel_bias", 0.0),
        ("softmax_scaled", "softmax_rel_bias", 0.0),
        ("softmax_canonical_scaled", "softmax_canonical_rel_bias", 0.0),
        ("softmax_unscaled", "softmax_rel_bias", 1.0),
        (
            "fixed_fosoftmax_unscaled",
            "fixed_forgetting_softmax_rel_bias",
            1.0,
        ),
        ("learned_fosoftmax_unscaled", "forgetting_softmax_rel_bias", 1.0),
        ("fixed_forgetting", "fixed_forgetting_rel_bias", 0.0),
        ("learned_forgetting", "forgetting_rel_bias", 0.0),
        ("local_forgetting_w64", "local_forgetting_rel_bias", 0.0),
        ("hybrid_forgetting_w64", "hybrid_forgetting_rel_bias", 0.0),
        ("taylor1", "taylor1_rel_bias", 0.0),
        ("taylor2", "taylor2_rel_bias", 0.0),
        ("additive_dot", "additive_dot", 0.0),
    ):
        torch.manual_seed(42)
        model = build(normalization, softmax_temperature).to(device).train()
        embeddings = model.get_item_embeddings(item_ids)
        output = model(
            past_lengths=lengths,
            past_ids=item_ids,
            past_embeddings=embeddings,
            past_payloads={"timestamps": timestamps},
        )
        feature_weights = torch.linspace(0.5, 1.5, D, device=device, dtype=output.dtype)
        loss = (output * feature_weights.view(1, 1, D)).mean()
        loss.backward()
        assert output.shape == (batch_size, padded_length, D)
        assert torch.isfinite(output).all()
        assert torch.isfinite(loss)

        forget_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "_forget_" in name
        ]
        if normalization in (
            "forgetting_rel_bias",
            "forgetting_softmax_rel_bias",
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            assert len(forget_grads) == 4
            assert all(grad is not None for grad in forget_grads)
            typed_forget_grads = [grad for grad in forget_grads if grad is not None]
            assert all(torch.isfinite(grad).all() for grad in typed_forget_grads)
            assert sum(grad.abs().sum() for grad in typed_forget_grads) > 0
        else:
            assert not forget_grads

        tail_gain_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "_hybrid_tail_rho" in name
        ]
        if normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            assert len(tail_gain_grads) == 2
            assert all(grad is not None for grad in tail_gain_grads)
            typed_tail_gain_grads = [
                grad for grad in tail_gain_grads if grad is not None
            ]
            assert all(torch.isfinite(grad).all() for grad in typed_tail_gain_grads)
            if normalization == "hybrid_forgetting_rel_bias":
                assert sum(grad.abs().sum() for grad in typed_tail_gain_grads) > 0
            else:
                assert sum(grad.abs().sum() for grad in typed_tail_gain_grads) == 0
        else:
            assert not tail_gain_grads

        if normalization == "forgetting_softmax_rel_bias":
            model.eval()
            with torch.no_grad():
                base_embeddings = embeddings.detach()
                _, cache = model.encode(
                    past_lengths=lengths,
                    past_ids=item_ids,
                    past_embeddings=base_embeddings,
                    past_payloads={"timestamps": timestamps},
                    return_cache_states=True,
                )
                dense_positions = lengths - 1
                jagged_offsets = torch.cat(
                    [
                        torch.zeros(1, device=device, dtype=lengths.dtype),
                        torch.cumsum(lengths, dim=0),
                    ]
                )
                jagged_positions = jagged_offsets[:-1] + dense_positions
                changed_embeddings = base_embeddings.clone()
                changed_embeddings[
                    torch.arange(batch_size, device=device), dense_positions
                ] += 0.1
                incremental = model.encode(
                    past_lengths=lengths,
                    past_ids=item_ids,
                    past_embeddings=changed_embeddings,
                    past_payloads={"timestamps": timestamps},
                    delta_x_offsets=(jagged_positions, dense_positions),
                    cache=cache,
                )
                fresh = model.encode(
                    past_lengths=lengths,
                    past_ids=item_ids,
                    past_embeddings=changed_embeddings,
                    past_payloads={"timestamps": timestamps},
                )
                torch.testing.assert_close(incremental, fresh)

        trainable = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{label}: shape={tuple(output.shape)} "
            f"loss={loss.item():.6f} params={trainable}"
        )
        del model, embeddings, output, loss
        gc.collect()
        torch.cuda.empty_cache()

    print("ALL OK")


if __name__ == "__main__":
    main()
