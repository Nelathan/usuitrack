import math
import sys
import types
import unittest
from unittest import mock

import torch

from usuitrack import UsuiTrack

from experiments.llm_synth_smoke import BackwardMemoryTrace, DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK, DEFAULT_MODEL, DEFAULT_SOURCE_HF_DATASET, build_parser, build_usuitrack_param_groups, install_state_matched_lora, maybe_compile_training_model, packed_text_limit, partition_usuitrack_params, repair_lfm2_gradient_checkpointing, select_trainable_params, validate_gradient_release_contract, validate_lora_optimizer_state_dtype, wandb_log
from experiments.llm_synth_smoke import apply_lr_schedule, cce_causal_lm_loss, gradient_norm_statistics, make_packed_batches, make_right_padded_batches, synth_masked_examples


def assert_param_membership(test_case, param, params, expected: bool) -> None:
    present = any(candidate is param for candidate in params)
    if expected:
        test_case.assertTrue(present)
    else:
        test_case.assertFalse(present)


class TinyTopology(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.conv = torch.nn.Conv1d(1, 2, 3)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class CountingCheckpointLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.sin()


class Lfm2Model(torch.nn.Module):
    def __init__(self, layers: int = 2):
        super().__init__()
        self.gradient_checkpointing = True
        self.layers = torch.nn.ModuleList([CountingCheckpointLayer() for _ in range(layers)])


class TinyLfmForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Lfm2Model()


class LlmHarnessParamScopeTest(unittest.TestCase):
    def test_compile_training_model_compiles_layers_in_place_only(self):
        model = TinyLfmForCausalLM()
        compiled = []

        def record_compile(layer, *args, **kwargs):
            compiled.append(layer)

        with mock.patch.object(torch.nn.Module, "compile", autospec=True, side_effect=record_compile):
            returned = maybe_compile_training_model(model, True)

        self.assertIs(returned, model)
        self.assertEqual(compiled, list(model.model.layers))
        self.assertEqual(model.model._usuitrack_compiled_layer_count, 2)

    def test_compile_training_model_disabled_leaves_layers_eager(self):
        model = TinyLfmForCausalLM()

        self.assertIs(maybe_compile_training_model(model, False), model)
        self.assertFalse(hasattr(model.model, "_usuitrack_compiled_layer_count"))

    def test_usuitrack_partition_leaves_only_matrices_in_matrix_optimizer(self):
        model = TinyTopology()
        named = [(name, param) for name, param in model.named_parameters()]

        matrix, fallback = partition_usuitrack_params(named)

        self.assertTrue(matrix)
        self.assertTrue(fallback)
        self.assertTrue(all(param.ndim == 2 for _name, param in matrix))
        self.assertTrue(all(param.ndim != 2 for _name, param in fallback))
        self.assertEqual({id(param) for _name, param in named}, {id(param) for _name, param in matrix + fallback})

    def test_state_matched_lora_targets_selected_matrices_and_keeps_bf16_adam_state(self):
        model = TinyTopology().to(dtype=torch.bfloat16)
        adapter_named, fallback_named, stats = install_state_matched_lora(
            model, "broad-no-embeddings", DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA
        )

        self.assertEqual(len(adapter_named), 2)
        self.assertEqual(sum(param.numel() for _name, param in adapter_named), DEFAULT_LORA_RANK * 8)
        self.assertTrue(all(param.dtype == torch.bfloat16 and param.requires_grad for _name, param in adapter_named))
        self.assertFalse(model.linear.base_layer.weight.requires_grad)
        self.assertTrue(fallback_named)
        self.assertTrue(all(param.requires_grad for _name, param in fallback_named))
        self.assertEqual(stats["selected_matrix_tensors"], 1)

        optimizer = torch.optim.AdamW(
            [param for _name, param in adapter_named],
            lr=2e-4,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
        )
        for _name, param in adapter_named:
            param.grad = torch.randn_like(param)
        optimizer.step()
        validate_lora_optimizer_state_dtype(optimizer, torch.bfloat16)
        self.assertTrue(all(state["exp_avg"].dtype == torch.bfloat16 for state in optimizer.state.values()))
        self.assertTrue(all(state["exp_avg_sq"].dtype == torch.bfloat16 for state in optimizer.state.values()))

    def test_backward_memory_trace_brackets_registered_preparation_hook(self):
        param = torch.nn.Parameter(torch.randn(3, 2))
        trace = BackwardMemoryTrace(torch.device("cpu"))
        trace.install_before([("weight", param)])
        middle = param.register_post_accumulate_grad_hook(lambda completed: setattr(completed, "grad", None))
        trace.install_after([("weight", param)])

        trace.start()
        param.square().sum().backward()
        trace.stop()

        self.assertEqual([event["phase"] for event in trace.events], ["before_prepare", "after_prepare"])
        self.assertTrue(trace.events[0]["grad_present"])
        self.assertFalse(trace.events[1]["grad_present"])
        self.assertGreater(trace.events[0]["grad_bytes"], 0)
        self.assertEqual(trace.events[1]["grad_bytes"], 0)
        self.assertEqual(trace.summary()["memory_trace_event_count"], 2)
        middle.remove()
        trace.close()

    def test_gradient_norm_statistics_describe_the_per_tensor_clip_population(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(1))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([10.0])

        global_norm, median, clipped_fraction = gradient_norm_statistics([first, second], clip_norm=6.0)

        torch.testing.assert_close(global_norm, torch.tensor(125.0).sqrt())
        torch.testing.assert_close(median, torch.tensor(5.0))
        torch.testing.assert_close(clipped_fraction, torch.tensor(0.5))

    def test_wandb_omits_unavailable_metrics_but_preserves_numeric_nan(self):
        class Run:
            def __init__(self):
                self.calls = []

            def log(self, data, *, step):
                self.calls.append((data, step))

        run = Run()
        wandb_log(run, {"eval/target_loss": 1.0, "eval/source_loss": None, "opt/numeric_problem": float("nan")}, step=10)

        self.assertEqual(len(run.calls), 1)
        data, step = run.calls[0]
        self.assertEqual(step, 10)
        self.assertEqual(data["eval/target_loss"], 1.0)
        self.assertNotIn("eval/source_loss", data)
        self.assertTrue(math.isnan(data["opt/numeric_problem"]))

    def test_cli_defaults_encode_current_baseline(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.param_scope, "broad-no-embeddings")
        self.assertEqual(args.seq_len, 1024)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.batching, "synth_right_padded_no_mask")
        self.assertEqual(args.max_steps, 1000)
        self.assertEqual(args.retention_hf_dataset, DEFAULT_SOURCE_HF_DATASET)
        self.assertEqual(args.rank_fraction, 0.1)
        self.assertEqual(args.usuitrack_lr, 1e-4)
        self.assertEqual(args.fallback_lr, 1e-4)
        self.assertEqual(args.lora_rank, 44)
        self.assertEqual(args.lora_alpha, 44)
        self.assertEqual(args.lora_lr, 2e-4)
        self.assertEqual(args.lora_grad_clip_norm, 1.0)
        self.assertEqual(args.lr_warmup_steps, 50)
        self.assertEqual(args.lr_final_ratio, 0.2)
        self.assertEqual(args.beta, 0.95)
        self.assertEqual(args.role_reads, [])
        self.assertEqual(args.basis_update_interval, 1)
        self.assertEqual(args.grad_clip_norm, 1.0)
        self.assertEqual(args.val_blocks, 8)
        self.assertEqual(args.retention_val_blocks, 8)
        self.assertEqual(args.wandb_log_every, 25)
        self.assertEqual(args.eval_every, 100)
        self.assertEqual(args.attn_implementation, "sdpa")
        self.assertTrue(args.activation_checkpointing)
        self.assertTrue(args.torch_compile)
        self.assertFalse(build_parser().parse_args(["--no-torch-compile"]).torch_compile)
        self.assertFalse(args.skip_validation)
        self.assertFalse(args.keep_grads_after_step)
        self.assertTrue(args.release_matrix_grads)
        self.assertFalse(build_parser().parse_args(["--no-release-matrix-grads"]).release_matrix_grads)
        self.assertFalse(args.trace_backward_memory)
        self.assertFalse(hasattr(args, "shadow_target_probe"))

    def test_lr_schedule_warms_up_then_decays_linearly_to_the_final_ratio(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)

        def lr_at(step, final_ratio=0.2):
            apply_lr_schedule(optimizer, [1e-4], step, 50, 1050, final_ratio)
            return optimizer.param_groups[0]["lr"]

        self.assertAlmostEqual(lr_at(25), 5e-5)
        self.assertAlmostEqual(lr_at(50), 1e-4)
        self.assertAlmostEqual(lr_at(550), 6e-5)
        self.assertAlmostEqual(lr_at(1050), 2e-5)
        self.assertAlmostEqual(lr_at(1050, final_ratio=1.0), 1e-4)

    def test_gradient_release_rejects_incompatible_harness_contracts(self):
        validate_gradient_release_contract(build_parser().parse_args([]))
        validate_gradient_release_contract(build_parser().parse_args(["--no-release-matrix-grads", "--grad-accum-steps", "2"]))
        for incompatible in (
            ["--grad-accum-steps", "2"],
            ["--keep-grads-after-step"],
        ):
            with self.assertRaises(ValueError):
                validate_gradient_release_contract(build_parser().parse_args(incompatible))

    def test_gradient_statistics_include_released_matrix_norms(self):
        live = torch.nn.Parameter(torch.zeros(3))
        live.grad = torch.tensor([3.0, 4.0, 0.0])
        total, median, clipped = gradient_norm_statistics([live], 4.0, (torch.tensor(12.0),))

        self.assertEqual(float(total), 13.0)
        self.assertEqual(float(median), 5.0)
        self.assertEqual(float(clipped), 1.0)

    def test_cli_has_no_loss_or_padding_option_garden(self):
        option_strings = {option for action in build_parser()._actions for option in action.option_strings}

        self.assertNotIn("--loss-impl", option_strings)
        self.assertNotIn("--chunked-lm-loss-tokens", option_strings)
        self.assertNotIn("--pad-to-max-length", option_strings)
        self.assertNotIn("--val-texts", option_strings)
        self.assertNotIn("--retention-val-texts", option_strings)
        self.assertNotIn("--print-shape-summary", option_strings)
        self.assertNotIn("--log-grad-norm", option_strings)
        self.assertNotIn("--log-norms", option_strings)

    def test_packed_text_limit_scales_with_requested_tokens(self):
        self.assertEqual(packed_text_limit(blocks=1, batch_size=1, seq_len=128), 16)
        self.assertEqual(packed_text_limit(blocks=4, batch_size=2, seq_len=1024), 16)
        self.assertEqual(packed_text_limit(blocks=200, batch_size=4, seq_len=1024), 1600)

    def test_broad_no_embeddings_includes_fallback_topology_without_embeddings_or_head(self):
        model = TinyTopology()

        trainable, stats = select_trainable_params(model, "broad-no-embeddings")

        assert_param_membership(self, model.linear.weight, trainable, True)
        assert_param_membership(self, model.linear.bias, trainable, True)
        assert_param_membership(self, model.norm.weight, trainable, True)
        assert_param_membership(self, model.norm.bias, trainable, True)
        assert_param_membership(self, model.conv.weight, trainable, True)
        assert_param_membership(self, model.embed_tokens.weight, trainable, False)
        assert_param_membership(self, model.lm_head.weight, trainable, False)
        self.assertGreater(stats["selected_matrix_params"], 0)
        self.assertGreater(stats["selected_fallback_params"], 0)
        self.assertGreater(stats["excluded_embedding_params"], 0)

    def test_matrices_no_embeddings_still_excludes_fallback_topology(self):
        model = TinyTopology()

        trainable, stats = select_trainable_params(model, "matrices-no-embeddings")

        assert_param_membership(self, model.linear.weight, trainable, True)
        assert_param_membership(self, model.linear.bias, trainable, False)
        assert_param_membership(self, model.norm.weight, trainable, False)
        assert_param_membership(self, model.conv.weight, trainable, False)
        assert_param_membership(self, model.embed_tokens.weight, trainable, False)
        self.assertEqual(stats["selected_fallback_params"], 0)
        self.assertGreater(stats["excluded_3d_params"], 0)

    def test_side_heuristic_tracks_the_d_model_axis(self):
        # d_model = 4 here (the axis every weight shares). up reads it on its
        # input (right) side, down writes it on its output (left) side, and the
        # square q_proj has no `out` in its name so it falls to the input side.
        up = torch.nn.Parameter(torch.randn(16, 4))
        down = torch.nn.Parameter(torch.randn(4, 16))
        q = torch.nn.Parameter(torch.randn(4, 4))
        named = [
            ("model.layers.0.mlp.up_proj.weight", up),
            ("model.layers.0.mlp.down_proj.weight", down),
            ("model.layers.0.self_attn.q_proj.weight", q),
        ]

        groups, stats = build_usuitrack_param_groups(named, rank_fraction=0.5)
        group_by_param = {id(param): group for group in groups for param in group["params"]}

        self.assertEqual(group_by_param[id(up)]["side"], "right")
        self.assertEqual(group_by_param[id(down)]["side"], "left")
        self.assertEqual(group_by_param[id(q)]["side"], "right")
        self.assertEqual(stats["side_policy_right_tensors"], 2)
        self.assertEqual(stats["side_policy_left_tensors"], 1)

    def test_side_override_and_out_name_force_the_output_side(self):
        # `out` in the name -> output side for a square weight; an override
        # entry sends a weight the d_model rule would place elsewhere.
        o = torch.nn.Parameter(torch.randn(4, 4))
        xk = torch.nn.Parameter(torch.randn(4, 8))
        named = [
            ("model.layers.0.self_attn.out_proj.weight", o),
            ("model.layers.0.cross_attn.to_k.weight", xk),
        ]

        groups, _ = build_usuitrack_param_groups(
            named, rank_fraction=0.5, side_overrides={"cross_attn.to_k": "in"}, d_model=4
        )
        group_by_param = {id(param): group for group in groups for param in group["params"]}

        self.assertEqual(group_by_param[id(o)]["side"], "left")
        self.assertEqual(group_by_param[id(xk)]["side"], "right")

    def test_lfm2_gradient_checkpointing_repair_wraps_decoder_layers(self):
        model = TinyLfmForCausalLM()

        wrapped = repair_lfm2_gradient_checkpointing(model)

        self.assertEqual(wrapped, 2)
        self.assertEqual(repair_lfm2_gradient_checkpointing(model), 0)
        self.assertTrue(all(getattr(layer, "_usuitrack_checkpoint_wrapped", False) for layer in model.model.layers))

        x = torch.randn(4, requires_grad=True)
        y = model.model.layers[0](x).sum()
        y.backward()

        self.assertEqual(model.model.layers[0].calls, 2)

    def test_lfm2_gradient_checkpointing_repair_uses_direct_forward_without_grad(self):
        model = TinyLfmForCausalLM()
        repair_lfm2_gradient_checkpointing(model)

        with torch.no_grad():
            model.model.layers[0](torch.randn(4))

        self.assertEqual(model.model.layers[0].calls, 1)

    def test_rank_fraction_clamps_to_half_the_smaller_side(self):
        first = torch.nn.Parameter(torch.randn(16, 4))
        second = torch.nn.Parameter(torch.randn(16, 4))
        named = [("a.up_proj.weight", first), ("b.up_proj.weight", second)]

        groups, stats = build_usuitrack_param_groups(named, rank_fraction=1.0)

        self.assertEqual(stats["effective_rank_min"], 2)
        self.assertEqual(stats["effective_rank_max"], 2)
        self.assertTrue(all(group["rank_fraction"] == 1.0 for group in groups))

    def test_rank_fraction_must_be_positive(self):
        named = [("a.up_proj.weight", torch.nn.Parameter(torch.randn(16, 4)))]

        with self.assertRaises(ValueError):
            build_usuitrack_param_groups(named, rank_fraction=0)

    def test_packed_batches_omit_attention_mask_for_sdpa_flash_path(self):
        class TokenizerStub:
            eos_token_id = 99

            def __call__(self, text, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("tokenizer stub called with unexpected kwargs")
                token_ids = [ord(char) % 10 for char in text]
                return {"input_ids": token_ids}

        batches = make_packed_batches(TokenizerStub(), ["abc", "def"], torch.device("cpu"), batch_size=2, seq_len=3, min_batches=1)

        self.assertEqual(len(batches), 1)
        self.assertIn("input_ids", batches[0])
        self.assertIn("labels", batches[0])
        self.assertNotIn("attention_mask", batches[0])
        self.assertTrue(torch.equal(batches[0]["input_ids"], batches[0]["labels"]))
        self.assertTrue((batches[0]["input_ids"] == 99).any())

    def test_right_padded_synth_batches_mask_context_without_attention_mask(self):
        class TokenizerStub:
            bos_token_id = 11
            eos_token_id = 99
            pad_token_id = 0

            vocab = {"Q": 1, "\n": 2, "-": 3, "A": 4, "B": 5}

            def __call__(self, text, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("tokenizer stub called with unexpected kwargs")
                return {"input_ids": [self.vocab[char] for char in text]}

        examples = synth_masked_examples(["Q\n\nAB"])
        batches = make_right_padded_batches(TokenizerStub(), examples, torch.device("cpu"), batch_size=1, seq_len=10, min_batches=1)

        self.assertNotIn("attention_mask", batches[0])
        self.assertTrue(torch.equal(batches[0]["input_ids"], torch.tensor([[11, 1, 2, 3, 3, 3, 2, 4, 5, 99]])))
        self.assertTrue(torch.equal(batches[0]["labels"], torch.tensor([[-100, -100, -100, -100, -100, -100, -100, 4, 5, 99]])))

    def test_cce_causal_lm_loss_uses_hidden_states_without_full_logits(self):
        captured = {}

        def fake_linear_cross_entropy(hidden_states, weight, targets, ignore_index, shift, reduction):
            captured["hidden_states"] = hidden_states
            captured["weight"] = weight
            captured["targets"] = targets
            captured["ignore_index"] = ignore_index
            captured["shift"] = shift
            return hidden_states.sum() * 0.0

        fake_module = types.ModuleType("cut_cross_entropy")
        fake_module.linear_cross_entropy = fake_linear_cross_entropy
        previous = sys.modules.get("cut_cross_entropy")
        sys.modules["cut_cross_entropy"] = fake_module
        try:
            class BaseStub(torch.nn.Module):
                def forward(self, input_ids, use_cache=False):
                    if use_cache:
                        raise AssertionError("base stub received unexpected cache flag")
                    hidden = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
                    return types.SimpleNamespace(last_hidden_state=hidden)

            class ModelStub(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = BaseStub()
                    self.lm_head = torch.nn.Linear(8, 4, bias=False)

            model = types.SimpleNamespace(_orig_mod=ModelStub())
            batch = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3]]),
            }

            loss = cce_causal_lm_loss(model, batch)
        finally:
            if previous is None:
                sys.modules.pop("cut_cross_entropy", None)
            else:
                sys.modules["cut_cross_entropy"] = previous

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(captured["ignore_index"], -100)
        self.assertTrue(captured["shift"])
        self.assertTrue(torch.equal(captured["targets"], torch.tensor([[-100, 2, 3]])))


if __name__ == "__main__":
    unittest.main()
