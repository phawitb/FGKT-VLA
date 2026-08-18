from pathlib import Path
import json
from hashlib import sha256
import sys

import pytest
from safetensors.torch import save_file
import torch

from fcut_vla.repair.continuation import (
    ContinuationHyperparameters,
    build_continuation_recipe,
)
from fcut_vla.repair.training import (
    ContinuationTrainingPlan,
    RepairTrainingError,
    materialize_candidate_policy_config,
    render_continuation_command,
    validate_artifact_separation,
    validate_training_runtime,
    verify_repair_candidate,
)
from fcut_vla.repair.pinned_train import main as pinned_train_main


def _recipe(source_digest="source-adapter-sha"):
    source_files = {
        "adapter_config.json": json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "modules_to_save": [],
                "target_modules": r"model\.expert\.q_proj",
                "base_model_name_or_path": "lerobot/smolvla_base",
                "revision": "base-sha",
            }
        ).encode(),
        "config.json": b"{}",
        "policy_preprocessor.json": b'{"kind":"pre"}',
        "policy_postprocessor.json": b'{"kind":"post"}',
        "policy_preprocessor_step_5_normalizer_processor.safetensors": b"normalizer",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": b"unnormalizer",
    }
    return build_continuation_recipe(
        source_run_hash="source-run-sha",
        source_adapter_sha256=source_digest,
        source_adapter_config_sha256=sha256(
            source_files["adapter_config.json"]
        ).hexdigest(),
        source_episode_sha256="episode-sha",
        failure_hash="failure-sha",
        task_alias="libero_spatial.task0",
        problem_id="pick_up_black_bowl",
        instruction="put the black bowl on the plate",
        dataset_repo="lerobot/libero_spatial_image",
        dataset_revision="dataset-sha",
        episode_indices=(8, 3),
        base_policy_repo="lerobot/smolvla_base",
        base_policy_revision="base-sha",
        lerobot_commit="lerobot-sha",
        peft_version="0.20.0",
        hf_libero_version="0.1.4",
        python_version="3.12.13",
        torch_version="2.11.0+cu130",
        device="cuda",
        tokenizer_repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        tokenizer_revision="tokenizer-sha",
        hyperparameters=ContinuationHyperparameters(20, 2, 1e-4, 2026),
        source_target_modules=("base_model.model.model.expert.q_proj",),
        source_policy_config_sha256=sha256(source_files["config.json"]).hexdigest(),
        source_preprocessor_sha256=sha256(
            source_files["policy_preprocessor.json"]
        ).hexdigest(),
        source_postprocessor_sha256=sha256(
            source_files["policy_postprocessor.json"]
        ).hexdigest(),
        source_processor_artifacts=tuple(
            sorted(
                (name, sha256(content).hexdigest())
                for name, content in source_files.items()
                if name.startswith(("policy_preprocessor", "policy_postprocessor"))
            )
        ),
    )


def _plan(tmp_path: Path, *, source_digest="source-adapter-sha"):
    checkpoint = tmp_path / "000010"
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    training_state.mkdir()
    (pretrained / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "modules_to_save": [],
                "target_modules": r"model\.expert\.q_proj",
                "base_model_name_or_path": "lerobot/smolvla_base",
                "revision": "base-sha",
            }
        )
    )
    (pretrained / "adapter_model.safetensors").write_text("{}")
    (pretrained / "config.json").write_bytes(b"{}")
    (pretrained / "policy_preprocessor.json").write_bytes(b'{"kind":"pre"}')
    (pretrained / "policy_postprocessor.json").write_bytes(b'{"kind":"post"}')
    (pretrained / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(
        b"normalizer"
    )
    (pretrained / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"unnormalizer"
    )
    (training_state / "training_step.json").write_text('{"step":10}')
    return ContinuationTrainingPlan(
        recipe=_recipe(source_digest),
        source_checkpoint_dir=checkpoint,
        source_pretrained_dir=pretrained,
        output_dir=tmp_path / "candidate",
        save_frequency=10,
        python_executable=Path(sys.executable),
    )


def test_continuation_command_resumes_source_and_filters_exact_episodes(tmp_path):
    plan = _plan(tmp_path)

    command = render_continuation_command(plan)

    assert command[:3] == (sys.executable, "-m", "fcut_vla.repair.pinned_train")
    assert "--fgkt-tokenizer-revision=tokenizer-sha" in command
    assert f"--policy.path={plan.source_pretrained_dir}" in command
    assert not any(argument.startswith("--config_path") for argument in command)
    assert "--resume=true" not in command
    assert "--dataset.episodes=[3,8]" in command
    assert "--dataset.repo_id=lerobot/libero_spatial_image" in command
    assert "--dataset.revision=dataset-sha" in command
    assert "--steps=20" in command
    assert "--batch_size=2" in command
    assert "--policy.optimizer_lr=0.0001" in command
    assert not any(argument.startswith("--optimizer.lr=") for argument in command)
    assert "--seed=2026" in command
    assert f"--output_dir={plan.output_dir}" in command
    assert "--policy.push_to_hub=false" in command
    assert "--policy.device=cuda" in command

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    parsed = SmolVLAConfig()
    parsed.optimizer_lr = float(
        next(argument.split("=", 1)[1] for argument in command if argument.startswith("--policy.optimizer_lr="))
    )
    assert parsed.get_optimizer_preset().lr == plan.recipe.hyperparameters.learning_rate


def test_pinned_train_injects_tokenizer_revision(monkeypatch):
    from transformers import AutoTokenizer
    import lerobot.scripts.lerobot_train as lerobot_train

    calls = []

    def fake_loader(cls, name, *args, **kwargs):
        calls.append((name, kwargs.get("revision")))
        return object()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_loader))
    monkeypatch.setattr(
        lerobot_train,
        "main",
        lambda: AutoTokenizer.from_pretrained(
            "tokenizer/repo", revision="transformers-internal-revision"
        ),
    )

    pinned_train_main(
        [
            "--fgkt-tokenizer-repo=tokenizer/repo",
            "--fgkt-tokenizer-revision=immutable-sha",
            "--",
            "--steps=1",
        ]
    )

    assert calls == [("tokenizer/repo", "immutable-sha")]


def test_training_plan_rejects_missing_verified_source_pretrained_artifact(tmp_path):
    plan = _plan(tmp_path)
    (plan.source_pretrained_dir / "adapter_model.safetensors").unlink()
    with pytest.raises(RepairTrainingError, match="source pretrained"):
        ContinuationTrainingPlan(**plan.__dict__)


def test_training_plan_rejects_source_policy_config_mutation(tmp_path):
    plan = _plan(tmp_path)
    (plan.source_pretrained_dir / "config.json").write_text('{"optimizer_lr":9}')

    with pytest.raises(RepairTrainingError, match="digest"):
        ContinuationTrainingPlan(**plan.__dict__)


def test_training_plan_rejects_source_adapter_config_mutation(tmp_path):
    plan = _plan(tmp_path)
    (plan.source_pretrained_dir / "adapter_config.json").write_text(
        '{"peft_type":"LORA","lora_dropout":0.9}'
    )

    with pytest.raises(RepairTrainingError, match="digest"):
        ContinuationTrainingPlan(**plan.__dict__)


def test_training_plan_rejects_source_processor_state_mutation(tmp_path):
    plan = _plan(tmp_path)
    state = (
        plan.source_pretrained_dir
        / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    state.write_bytes(b"mutated")

    with pytest.raises(RepairTrainingError, match="processor artifact manifest"):
        ContinuationTrainingPlan(**plan.__dict__)


def test_training_plan_rejects_output_inside_source_checkpoint(tmp_path):
    plan = _plan(tmp_path)

    with pytest.raises(RepairTrainingError, match="outside source checkpoint"):
        ContinuationTrainingPlan(
            **{
                **plan.__dict__,
                "output_dir": plan.source_checkpoint_dir / "candidate",
            }
        )


def test_candidate_run_and_source_run_must_not_overlap(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    validate_artifact_separation(source, tmp_path / "candidate-run")

    with pytest.raises(RepairTrainingError, match="overlap"):
        validate_artifact_separation(source, source / "candidate")
    with pytest.raises(RepairTrainingError, match="overlap"):
        validate_artifact_separation(source / "nested", source)


def test_training_runtime_requires_exact_clean_lerobot_and_peft_versions():
    assert validate_training_runtime(
        expected_lerobot_commit="lerobot-sha",
        expected_peft_version="0.20.0",
        actual_lerobot_commit="lerobot-sha",
        actual_peft_version="0.20.0",
        lerobot_dirty=False,
    ) == {"lerobot_commit": "lerobot-sha", "peft_version": "0.20.0"}

    with pytest.raises(RepairTrainingError, match="runtime provenance"):
        validate_training_runtime(
            expected_lerobot_commit="lerobot-sha",
            expected_peft_version="0.20.0",
            actual_lerobot_commit="other-sha",
            actual_peft_version="0.20.0",
            lerobot_dirty=False,
        )
    with pytest.raises(RepairTrainingError, match="dirty"):
        validate_training_runtime(
            expected_lerobot_commit="lerobot-sha",
            expected_peft_version="0.20.0",
            actual_lerobot_commit="lerobot-sha",
            actual_peft_version="0.20.0",
            lerobot_dirty=True,
        )


def _write_candidate(plan, *, value=2.0, targets=None, configured_targets=None):
    checkpoint = (
        plan.output_dir
        / "checkpoints"
        / f"{plan.recipe.hyperparameters.steps:06d}"
        / "pretrained_model"
    )
    checkpoint.mkdir(parents=True)
    target_names = targets or ("base_model.model.model.expert.q_proj",)
    tensors = {}
    for target in target_names:
        tensors[f"{target}.lora_A.default.weight"] = torch.full((1, 1), value)
        tensors[f"{target}.lora_B.default.weight"] = torch.full((1, 1), value)
    save_file(tensors, checkpoint / "adapter_model.safetensors")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "modules_to_save": [],
                "target_modules": configured_targets or r"model\.expert\.q_proj",
                "base_model_name_or_path": "lerobot/smolvla_base",
                "revision": "base-sha",
            }
        )
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"optimizer_lr": plan.recipe.hyperparameters.learning_rate})
    )
    (checkpoint / "policy_preprocessor.json").write_bytes(b'{"kind":"pre"}')
    (checkpoint / "policy_postprocessor.json").write_bytes(b'{"kind":"post"}')
    (checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(
        b"normalizer"
    )
    (checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"unnormalizer"
    )
    training_state = checkpoint.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(
        json.dumps({"step": plan.recipe.hyperparameters.steps})
    )
    log = plan.output_dir / "stderr.log"
    log.write_text(
        "num_learnable_params=100 (100)\nnum_total_params=100000 (100K)\n"
    )
    return log


def _source_evidence(plan, *, digest="source-adapter-sha"):
    return {
        "adapter_sha256": digest,
        "rank": 8,
        "alpha": 16,
        "trainable_parameters": 100,
        "total_parameters": 100_000,
        "resolved_targets": ["base_model.model.model.expert.q_proj"],
    }


def test_candidate_verification_requires_new_digest_and_same_lora_contract(tmp_path):
    plan = _plan(tmp_path)
    log = _write_candidate(plan)

    report = verify_repair_candidate(
        plan=plan,
        source_evidence=_source_evidence(plan),
        candidate_log=log,
    )

    assert report.source_adapter_sha256 == "source-adapter-sha"
    assert report.candidate_adapter_sha256 != report.source_adapter_sha256
    assert report.resolved_targets == (
        "base_model.model.model.expert.q_proj",
    )
    assert report.completed_step == 20

    state = (
        plan.output_dir
        / "checkpoints"
        / "000020"
        / "pretrained_model"
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    state.write_bytes(b"mutated-after-completion")
    with pytest.raises(RepairTrainingError, match="processor artifact manifest"):
        verify_repair_candidate(
            plan=plan,
            source_evidence=_source_evidence(plan),
            candidate_log=log,
        )


def test_materialize_candidate_policy_config_for_peft_path_checkpoint(tmp_path):
    plan = _plan(tmp_path)
    checkpoint = (
        plan.output_dir
        / "checkpoints"
        / "000020"
        / "pretrained_model"
    )
    checkpoint.mkdir(parents=True)

    path = materialize_candidate_policy_config(plan)

    assert path == checkpoint / "config.json"
    assert json.loads(path.read_text()) == {
        "device": "cuda",
        "optimizer_lr": 1e-4,
    }


def test_candidate_verification_rejects_same_digest_or_target_drift(tmp_path):
    plan = _plan(tmp_path)
    log = _write_candidate(plan)
    first = verify_repair_candidate(
        plan=plan,
        source_evidence=_source_evidence(plan),
        candidate_log=log,
    )
    same = _plan(tmp_path / "same", source_digest=first.candidate_adapter_sha256)
    same_log = _write_candidate(same)
    with pytest.raises(RepairTrainingError, match="distinct adapter digest"):
        verify_repair_candidate(
            plan=same,
            source_evidence=_source_evidence(
                same, digest=first.candidate_adapter_sha256
            ),
            candidate_log=same_log,
        )

    other = _plan(tmp_path / "other")
    other_log = _write_candidate(
        other, targets=("base_model.model.model.expert.k_proj",)
    )
    with pytest.raises(RepairTrainingError, match="target contract"):
        verify_repair_candidate(
            plan=other,
            source_evidence=_source_evidence(other),
            candidate_log=other_log,
        )

    broad = _plan(tmp_path / "broad")
    broad_log = _write_candidate(broad, configured_targets=".*")
    with pytest.raises(RepairTrainingError, match="target configuration"):
        verify_repair_candidate(
            plan=broad,
            source_evidence=_source_evidence(broad),
            candidate_log=broad_log,
        )
