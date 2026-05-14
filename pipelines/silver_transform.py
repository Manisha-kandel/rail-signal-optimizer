import os
import sys

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from delta import configure_spark_with_delta_pip

BRONZE_PATH = "./data/bronze/train_events"
SILVER_PATH = "./data/silver/train_events"
CHECKPOINT  = "./data/checkpoints/silver"


def build_session() -> SparkSession:
    builder = (
        SparkSession.builder  # type: ignore[attr-defined]
        .appName("RailSignal-Silver")
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

    query = (
        silver.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT)
        .start(SILVER_PATH)
    )

    print(f"Silver transform running: Bronze → {SILVER_PATH}")
    print("Ctrl+C to stop.\n")
    query.awaitTermination()


if __name__ == "__main__":
    run()
