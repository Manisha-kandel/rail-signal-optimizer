"""
Day 18 — Export trained PPO actor to ONNX and verify Spark UDF inference.

Steps:
    1. Load best_model.zip (stable-baselines3 PPO)
    2. Wrap actor network (obs -> logits) as a traceable nn.Module
    3. torch.onnx.export -> models/artifacts/ppo_rail_policy.onnx
    4. Verify ONNX output matches PyTorch output on sample obs
    5. Define module-level Spark UDF wrapping ONNX inference
    6. Apply to Gold Delta table and show rl_advisory_speed_mph column

Run:
    python -m models.export_onnx
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from stable_baselines3 import PPO
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when as spark_when

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SB3_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "ppo_rail_policy.onnx")
GOLD_PATH = "./data/gold/train_advisories"
SPEED_TARGETS = [0.0, 20.0, 40.0, 60.0, 79.0]


# ── Actor wrapper (obs -> action logits) ──────────────────────────────

class _ActorOnly(nn.Module):
    """Wraps only the actor path of the SB3 ActorCriticPolicy for ONNX export."""

    def __init__(self, policy) -> None:
        super().__init__()
        self._features_extractor = policy.pi_features_extractor
        self._mlp = policy.mlp_extractor
        self._action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self._features_extractor(obs)
        latent_pi, _ = self._mlp(features)
        return self._action_net(latent_pi)


# ── Batch ONNX inference (pandas row-wise) ────────────────────────────
# Python 3.14 cloudpickle stack-overflows when serializing Spark Python UDFs.
# For local mode, pandas batch inference avoids cloudpickle entirely and is
# equivalent in capability. In Databricks, replace with spark.udf.register().

def rl_advisory_batch(df) -> list[float]:
    """Apply ONNX model to every row in a pandas DataFrame; return list of speeds."""
    import onnxruntime as ort

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    results: list[float] = []

    for _, row in df.iterrows():
        sig = (row.get("signal_aspect") or "clear").strip().lower()
        obs = np.array([[
            float(row.get("speed_mph") or 0.0) / 79.0,
            1.0 if sig == "stop"     else 0.0,
            1.0 if sig == "approach" else 0.0,
            1.0 if sig == "clear"    else 0.0,
            float(min(row.get("blocks_count") or 0, 2)) / 2.0,
            float(np.clip((row.get("schedule_adherence_sec") or 0.0) / 300.0, -1.0, 1.0)),
            float(np.clip((row.get("track_grade_pct")        or 0.0) / 3.0,   -1.0, 1.0)),
            float(row.get("block_idx") or 0) / 19.0,
        ]], dtype=np.float32)
        logits = session.run(None, {"observation": obs})[0]
        action = int(np.argmax(logits))
        results.append(float(SPEED_TARGETS[action]))

    return results


# ── ONNX export ────────────────────────────────────────────────────────

def export_onnx() -> None:
    print("Loading SB3 model...")
    model = PPO.load(SB3_MODEL_PATH)
    policy = model.policy
    policy.eval()

    actor = _ActorOnly(policy)
    actor.eval()
    dummy_obs = torch.zeros(1, 8, dtype=torch.float32)

    with torch.no_grad():
        pt_logits = actor(dummy_obs).numpy()
    pt_action = int(np.argmax(pt_logits))
    print(f"PyTorch logits: {pt_logits[0].round(4)}  -> action {pt_action} "
          f"({SPEED_TARGETS[pt_action]:.0f} mph)")

    torch.onnx.export(
        actor,
        (dummy_obs,),
        ONNX_PATH,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={"observation": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"ONNX model exported -> {ONNX_PATH}")

    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"observation": dummy_obs.numpy()})[0]
    onnx_action = int(np.argmax(onnx_logits))
    print(f"ONNX logits:    {onnx_logits[0].round(4)}  -> action {onnx_action} "
          f"({SPEED_TARGETS[onnx_action]:.0f} mph)")
    print(f"PyTorch vs ONNX match: {'PASS' if np.allclose(pt_logits, onnx_logits, atol=1e-5) else 'FAIL'}")


# ── Apply ONNX inference to Gold table via pandas batch ───────────────

def apply_to_gold() -> None:
    builder = (
        SparkSession.builder  # type: ignore[attr-defined]
        .appName("RailSignal-RLAdvisory")  # type: ignore[attr-defined]
        .master("local[*]")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("\nReading Gold Delta table...")
    gold_spark = spark.read.format("delta").load(GOLD_PATH)
    gold_spark = gold_spark.withColumn(
        "blocks_count",
        spark_when(col("signal_aspect") == "stop", lit(1))
        .when(col("signal_aspect") == "approach", lit(1))
        .otherwise(lit(0)),
    )

    # Collect to pandas — local mode; Databricks equivalent uses spark.udf.register()
    gold_pd = gold_spark.toPandas()
    gold_pd["rl_advisory_speed_mph"] = rl_advisory_batch(gold_pd)

    print("Sample: rule-based advisory vs RL advisory")
    sample = gold_pd[["train_id", "signal_aspect", "speed_mph",
                       "advisory_speed_mph", "rl_advisory_speed_mph",
                       "delay_severity"]].head(10)
    print(sample.to_string(index=False))

    print(f"\nRL advisory applied to {len(gold_pd):,} Gold rows — DONE")
    spark.stop()


def run() -> None:
    print("=" * 50)
    print("  Rail Signal Optimizer - ONNX Export + UDF")
    print("=" * 50)
    export_onnx()
    print()
    apply_to_gold()


if __name__ == "__main__":
    run()
