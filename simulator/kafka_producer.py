import time
import sys
from confluent_kafka import KafkaError, Message, Producer
from simulator.train_generator import TrainState, simulate_step, TRAIN_IDS

KAFKA_BROKER = "localhost:9092"
TOPIC = "train_events"

def delivery_report(err: KafkaError | None, msg: Message) -> None:
    if err is not None:
        print(f"[ERROR] Delivery failed for {msg.key()}: {err}", file=sys.stderr)

def run() -> None:
    producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    states = [TrainState(tid, i * 3) for i, tid in enumerate(TRAIN_IDS)]

    print(f"Producer running → topic '{TOPIC}'. Ctrl+C to stop.\n")
    try:
        while True:
            events = simulate_step(states)
            for event in events:
                producer.produce(
                    topic=TOPIC,
                    key=event.train_id,
                    value=event.to_kafka_bytes(),
                    callback=delivery_report,
                )
            producer.poll(0)
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        print("Producer shut down cleanly.")

if __name__ == "__main__":
    run()