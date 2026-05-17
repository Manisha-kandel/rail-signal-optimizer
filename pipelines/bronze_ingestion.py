import os
import sys

# Must be set before JVM starts — winutils.exe shim for Windows filesystem ops
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    IntegerType, ArrayType,
)
from delta import configure_spark_with_delta_pip

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC        = "train_events"
BRONZE_PATH  = "./data/bronze/train_events"
CHECKPOINT   = "./data/checkpoints/bronze"

BLOCK_AHEAD_SCHEMA = StructType([
    StructField("block_id",                StringType()),
    StructField("occupying_train",         StringType()),
    StructField("estimated_clearance_sec", FloatType()),
])

TRAIN_EVENT_SCHEMA = StructType([
    StructField("event_id",                  StringType()),
    StructField("timestamp_utc",             StringType()),
    StructField("train_id",                  StringType()),
    StructField("block_id",                  StringType()),
    StructField("block_idx",                 IntegerType()),
    StructField("subdivision",               StringType()),
    StructField("current_speed_mph",         FloatType()),
    StructField("max_authorized_speed_mph",  FloatType()),
    StructField("gross_tonnage",             IntegerType()),
    StructField("signal_aspect_ahead",       StringType()),
    StructField("blocks_ahead_occupied",     ArrayType(BLOCK_AHEAD_SCHEMA)),
    StructField("schedule_adherence_sec",    FloatType()),
    StructField("track_grade_pct",           FloatType()),
])


def build_session() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("RailSignal-Bronze")  # type: ignore[attr-defined]
        .master("local[*]")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"],
    ).getOrCreate()


def run() -> None:
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw
        .select(from_json(col("value").cast("string"), TRAIN_EVENT_SCHEMA).alias("d"))
        .select("d.*")
    )

    query = (
        parsed.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT)
        .start(BRONZE_PATH)
    )

    print(f"Bronze ingestion running → {BRONZE_PATH}")
    print("Ctrl+C to stop.\n")
    query.awaitTermination()


if __name__ == "__main__":
    run()
