# Architecture — Real-Time Rail Signal & Scheduling Optimizer

---

## 1. Data Flow Overview

```
Train Simulator (Python)
    │  TrainEvent (Pydantic v2, JSON)
    ▼
Kafka Producer ──► topic: train_events ──► Streamlit Consumer (live map)
                                      │
                                      └──► Spark Structured Streaming
                                                │
                                          Bronze Delta Table (raw)
                                                │
                                          Silver Delta Table (validated, stateful join)
                                                │  shockwave UDF
                                          Gold Delta Table (speed advisory, delay estimate)
                                                │
                                          PPO Agent (Gymnasium → stable-baselines3)
                                                │  ONNX export
                                          Spark UDF ──► Gold advisory column
                                                │
                                          Streamlit Dashboard (Week 4 rebuild)
```

---

## 2. Simulator Layer

**Files:** `simulator/train_event.py`, `simulator/train_generator.py`, `simulator/kafka_producer.py`

### TrainEvent Schema (Pydantic v2)

```
TrainEvent
├── event_id          uuid4
├── timestamp_utc     ISO-8601
├── train_id          T_001 … T_005
├── block_id          B_001 … B_020
├── block_idx         int 0-19
├── subdivision       "Harrisburg_Sub"
├── current_speed_mph float
├── max_authorized_speed_mph  79.0
├── signal_aspect_ahead       clear | approach | stop
├── blocks_ahead_occupied     list[BlockAhead]
├── schedule_adherence_sec    float  (negative = late)
└── track_grade_pct           float
```

### Physics Model

Each 2-second tick per train:

| Signal Aspect | Speed Update |
|---|---|
| stop | `max(0, speed − U(8, 15))` |
| approach | `max(20, speed − U(2, 6))` |
| clear | `min(79, speed + U(1, 4))` — advance block with p=0.5 |

### Signal Logic (3-aspect block system)

```python
if block_idx+1 (mod 20) is occupied → STOP
elif block_idx+2 (mod 20) is occupied → APPROACH
else → CLEAR
```

Circular track (modular arithmetic) eliminates end-of-line boundary conditions.

---

## 3. Streaming Layer

**Files:** `docker-compose.yml`

Single-broker Kafka on `localhost:9092`. Topic `train_events` auto-created.
Zookeeper handles broker coordination internally (not exposed to host).

Producer key = `train_id` → same-train events always land on the same partition → ordering guaranteed per train.

---

## 4. Processing Layer (Week 2)

**Files:** `pipelines/bronze_ingestion.py`, `pipelines/silver_transform.py`, `pipelines/gold_aggregation.py`

### Medallion Architecture

```
Bronze   Raw JSON bytes from Kafka, append-only, schema-on-read
           ↓  parse · validate · watermark (event_time, 30s lateness)
Silver   Typed columns, stateful block-occupancy join (StreamStream)
           ↓  shockwave UDF · delay estimate
Gold     Speed advisory column, RL inference column, aggregated KPIs
```

### Shockwave UDF (planned)

For each train, look back N upstream trains. If lead train is at STOP:
```
delay_cascade[i] = headway_gap[i] × (speed_drop[i+1] / speed_nominal)
```
Modeled as a directed graph walk on block adjacency — each node is a block, edge weight is clearance time.

---

## 5. Optimization Layer (Week 3)

**Files:** `models/rl_environment.py`, `models/train_ppo.py`, `models/evaluate_policy.py`

### MDP Formulation

| Element | Definition |
|---|---|
| State | `(speed_norm, signal_one_hot[3], blocks_ahead[0..2], adherence_norm, grade)` |
| Action | Discrete speed advisory: {-10, -5, 0, +5, +10} mph delta |
| Reward | `−|delay| − α·|braking_force|` |
| Episode | One train traversal of 20-block corridor |
| Horizon | 200 steps |

### Training Pipeline

```
Gymnasium env → PPO (stable-baselines3, 500k steps, Colab T4)
    → evaluate vs rule-based baseline
    → serialize to ONNX
    → wrap as Spark UDF
    → write speed_advisory to Gold table
```

---

## 6. Dashboard Layer

**Week 1 (current):** Streamlit reads directly from Kafka consumer. Displays:
- Live 20-block track map (color = signal aspect)
- Train advisory feed (speed, signal, schedule adherence)
- Refresh cadence: ~2s via `st.cache_resource` consumer + `st.rerun()`

**Week 4 (planned):** Rebuild to read from Gold Delta table. Adds:
- Plotly Marey (time-distance) diagram
- RL advisory overlay (purple = advisory active)
- Deploy to Streamlit Cloud

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| Pydantic v2 `model_dump_json()` | Kafka message = schema-validated JSON; same schema consumed by Spark |
| Kafka key = `train_id` | Per-train ordering; enables stateful join in Spark without shuffle |
| `@st.cache_resource` for Consumer | Confluent C-extension not hashable by Streamlit session state |
| Circular track (`% len(BLOCKS)`) | Eliminates boundary conditions; trains loop continuously for training data |
| ONNX for model serving | Single artifact deployable in Spark UDF without Python ML dependencies on workers |
| Databricks Community Edition | Free tier; forces restartable pipeline design (no always-on clusters) |
