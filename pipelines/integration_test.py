"""
Day 13 — End-to-end integration test
Validates the full Bronze → Silver → Gold medallion pipeline
by reading each Delta table as batch and running data quality checks.
No Kafka/Docker required — reads from local Delta files.
"""
import os
import sys

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, DataFrame
from delta import configure_spark_with_delta_pip

BRONZE_PATH    = "./data/bronze/train_events"
SILVER_PATH    = "./data/silver/train_events"
OCCUPANCY_PATH = "./data/silver/block_occupancy"
GOLD_PATH      = "./data/gold/train_advisories"

PASS = "  PASS"
FAIL = "  FAIL"


def build_session() -> SparkSession:
    builder = (
        SparkSession.builder  # type: ignore[attr-defined]
        .appName("RailSignal-IntegrationTest")  # type: ignore[attr-defined]
        .master("local[2]")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def check(label: str, passed: bool) -> bool:
    print(f"{'[PASS]' if passed else '[FAIL]'} {label}")
    return passed


def validate_bronze(spark: SparkSession) -> int:
    print("\n── Bronze ──────────────────────────────────────")
    df: DataFrame = spark.read.format("delta").load(BRONZE_PATH)
    count = df.count()
    failures = 0

    failures += 0 if check(f"Row count > 0 ({count:,})", count > 0) else 1
    failures += 0 if check(
        "Required columns present",
        all(c in df.columns for c in
            ["train_id", "block_id", "signal_aspect_ahead",
             "current_speed_mph", "schedule_adherence_sec"])
    ) else 1
    null_train = df.filter(df.train_id.isNull()).count()
    failures += 0 if check(f"No null train_id ({null_train})", null_train == 0) else 1

    print(f"  Sample:")
    df.select("train_id", "block_id", "signal_aspect_ahead",
              "current_speed_mph").show(3, truncate=False)
    return failures


def validate_silver(spark: SparkSession) -> int:
    print("\n── Silver (validated events) ───────────────────")
    df: DataFrame = spark.read.format("delta").load(SILVER_PATH)
    count = df.count()
    failures = 0

    failures += 0 if check(f"Row count > 0 ({count:,})", count > 0) else 1
    failures += 0 if check(
        "event_time column present",
        "event_time" in df.columns
    ) else 1

    bad_signal = df.filter(
        ~df.signal_aspect_ahead.isin("clear", "approach", "stop")
    ).count()
    failures += 0 if check(
        f"All signal aspects valid ({bad_signal} invalid)", bad_signal == 0
    ) else 1

    bad_speed = df.filter(
        (df.current_speed_mph < 0) | (df.current_speed_mph > 79)
    ).count()
    failures += 0 if check(
        f"All speeds in 0-79 mph ({bad_speed} out of range)", bad_speed == 0
    ) else 1

    print(f"  Sample:")
    df.select("train_id", "block_id", "signal_aspect_ahead",
              "current_speed_mph", "event_time").show(3, truncate=False)
    return failures


def validate_occupancy(spark: SparkSession) -> int:
    print("\n── Silver (block occupancy state) ──────────────")
    df: DataFrame = spark.read.format("delta").load(OCCUPANCY_PATH)
    count = df.count()
    failures = 0

    failures += 0 if check(f"Row count > 0 ({count:,})", count > 0) else 1
    failures += 0 if check(
        "Window columns present",
        all(c in df.columns for c in ["window_start", "window_end", "train_id"])
    ) else 1
    train_count = df.select("train_id").distinct().count()
    failures += 0 if check(
        f"All 5 trains represented ({train_count})", train_count == 5
    ) else 1

    print(f"  Sample:")
    df.select("window_start", "train_id", "block_id",
              "signal_aspect", "speed_mph").show(3, truncate=False)
    return failures


def validate_gold(spark: SparkSession) -> int:
    print("\n── Gold (advisories) ───────────────────────────")
    df: DataFrame = spark.read.format("delta").load(GOLD_PATH)
    count = df.count()
    failures = 0

    failures += 0 if check(f"Row count > 0 ({count:,})", count > 0) else 1
    failures += 0 if check(
        "Gold columns present",
        all(c in df.columns for c in
            ["shockwave_delay_sec", "advisory_speed_mph", "delay_severity"])
    ) else 1

    bad_severity = df.filter(
        ~df.delay_severity.isin("nominal", "moderate", "critical")
    ).count()
    failures += 0 if check(
        f"All severity values valid ({bad_severity} invalid)", bad_severity == 0
    ) else 1

    bad_advisory = df.filter(
        (df.advisory_speed_mph < 0) | (df.advisory_speed_mph > 79)
    ).count()
    failures += 0 if check(
        f"Advisory speeds in 0-79 mph ({bad_advisory} out of range)",
        bad_advisory == 0
    ) else 1

    print(f"  Sample:")
    df.select("train_id", "signal_aspect", "shockwave_delay_sec",
              "advisory_speed_mph", "delay_severity").show(5, truncate=False)
    return failures


def run() -> None:
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 52)
    print("  Rail Signal Optimizer — Integration Test")
    print("  Pipeline: Kafka → Bronze → Silver → Gold")
    print("=" * 52)

    total_failures = 0
    total_failures += validate_bronze(spark)
    total_failures += validate_silver(spark)
    total_failures += validate_occupancy(spark)
    total_failures += validate_gold(spark)

    print("\n" + "=" * 52)
    if total_failures == 0:
        print("  ALL CHECKS PASSED — pipeline validated end-to-end")
    else:
        print(f"  {total_failures} CHECK(S) FAILED")
    print("=" * 52)

    spark.stop()


if __name__ == "__main__":
    run()
