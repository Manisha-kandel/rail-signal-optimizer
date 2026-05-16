import os
import sys

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, greatest, least, abs as spark_abs,
)
from delta import configure_spark_with_delta_pip

OCCUPANCY_PATH = "./data/silver/block_occupancy"
GOLD_PATH      = "./data/gold/train_advisories"
CHECKPOINT     = "./data/checkpoints/gold"


def build_session() -> SparkSession:
    builder = (
        SparkSession.builder  # type: ignore[attr-defined]
        .appName("RailSignal-Gold")  # type: ignore[attr-defined]
        .master("local[*]")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def run() -> None:
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    occupancy = (
        spark.readStream
        .format("delta")
        .load(OCCUPANCY_PATH)
    )

    # base_lateness: magnitude of delay for late trains only (0 if on-time/early)
    base_lateness = spark_abs(least(col("schedule_adherence_sec"), lit(0.0)))

    # speed_factor: 0.0 at 79 mph (full speed), 1.0 at standstill — slower = bigger wave
    speed_factor = greatest(
        lit(0.0),
        (lit(79.0) - col("speed_mph")) / lit(79.0),
    )

    # ── Shockwave propagation (native Spark SQL — no Python UDF, runs in JVM) ──
    shockwave_delay = (
        when(col("signal_aspect") == "stop",
             base_lateness + lit(120.0) + speed_factor * lit(60.0))
        .when(col("signal_aspect") == "approach",
              base_lateness * lit(0.5) + lit(30.0) + speed_factor * lit(20.0))
        .otherwise(base_lateness * lit(0.1))
    )

    # ── Speed advisory (rule-based; replaced by RL ONNX UDF in Week 3) ─────────
    advisory_speed = (
        when(col("signal_aspect") == "stop",     lit(0.0))
        .when(col("signal_aspect") == "approach", col("speed_mph") * lit(0.6))
        .otherwise(lit(79.0))
    )

    # ── Delay severity flag ──────────────────────────────────────────────────────
    delay_severity = (
        when(col("shockwave_delay_sec") > 180, lit("critical"))
        .when(col("shockwave_delay_sec") > 60,  lit("moderate"))
        .otherwise(lit("nominal"))
    )

    gold = (
        occupancy
        .withColumn("shockwave_delay_sec", shockwave_delay)
        .withColumn("advisory_speed_mph",  advisory_speed)
        .withColumn("delay_severity",      delay_severity)
    )

    query = (
        gold.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT)
        .start(GOLD_PATH)
    )

    print(f"Gold aggregation running: Silver → {GOLD_PATH}")
    print("Columns: shockwave_delay_sec | advisory_speed_mph | delay_severity")
    print("Ctrl+C to stop.\n")
    query.awaitTermination()


if __name__ == "__main__":
    run()
