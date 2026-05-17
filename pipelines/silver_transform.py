import os
import sys

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, window, last
from delta import configure_spark_with_delta_pip

BRONZE_PATH    = "./data/bronze/train_events"
SILVER_PATH    = "./data/silver/train_events"
OCCUPANCY_PATH = "./data/silver/block_occupancy"
CHECKPOINT     = "./data/checkpoints/silver"
CHECKPOINT_OCC = "./data/checkpoints/silver_occupancy"


def build_session() -> SparkSession:
    builder = (
        SparkSession.builder  # type: ignore[attr-defined]
        .appName("RailSignal-Silver")  # type: ignore[attr-defined]
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

    bronze = (
        spark.readStream
        .format("delta")
        .load(BRONZE_PATH)
    )

    silver = (
        bronze
        # Parse ISO-8601 string → proper timestamp for watermarking
        .withColumn("event_time", to_timestamp(col("timestamp_utc")))
        # Watermark: tolerate up to 30s of late-arriving events
        .withWatermark("event_time", "30 seconds")
        # Drop records with null key fields
        .filter(col("train_id").isNotNull())
        .filter(col("block_id").isNotNull())
        .filter(col("event_time").isNotNull())
        # Validate signal aspect is a known value
        .filter(col("signal_aspect_ahead").isin("clear", "approach", "stop"))
        # Validate speed within Harrisburg Sub operating bounds
        .filter(
            (col("current_speed_mph") >= 0) &
            (col("current_speed_mph") <= 79.0)
        )
        # Validate block index in corridor range
        .filter(
            (col("block_idx") >= 0) &
            (col("block_idx") < 20)
        )
    )

    # ── Query 1: validated Silver events (append) ─────────────────
    query1 = (
        silver.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT)
        .start(SILVER_PATH)
    )

    # ── Query 2: stateful block occupancy state (update) ───────────
    # 30s sliding window, 10s slide — latest position per train
    occupancy = (
        silver
        .groupBy(
            window(col("event_time"), "30 seconds", "10 seconds"),
            col("train_id"),
        )
        .agg(
            last("block_id").alias("block_id"),
            last("block_idx").alias("block_idx"),
            last("signal_aspect_ahead").alias("signal_aspect"),
            last("current_speed_mph").alias("speed_mph"),
            last("schedule_adherence_sec").alias("schedule_adherence_sec"),
            last("track_grade_pct").alias("track_grade_pct"),
        )
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("train_id"),
            col("block_id"),
            col("block_idx"),
            col("signal_aspect"),
            col("speed_mph"),
            col("schedule_adherence_sec"),
            col("track_grade_pct"),
        )
    )

    query2 = (
        occupancy.writeStream
        .format("delta")
        .outputMode("complete")
        .option("checkpointLocation", CHECKPOINT_OCC)
        .start(OCCUPANCY_PATH)
    )

    print(f"Silver transform running: Bronze → {SILVER_PATH}")
    print(f"Block occupancy state  : Silver → {OCCUPANCY_PATH}")
    print("Ctrl+C to stop.\n")

    query2.awaitTermination()
    query1.awaitTermination()


if __name__ == "__main__":
    run()
