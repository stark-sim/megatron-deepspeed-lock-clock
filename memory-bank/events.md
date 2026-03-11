# Events

## Event Architecture
The project is not built around a message broker, but experiment execution still has an important event flow through tracker artifacts and logs.

## Event Catalog
| Event Name | Producer | Consumer(s) | Payload |
|------------|----------|-------------|---------|
| `initialized` | Experiment tracker | Human/agent analysis | Run metadata, command hash, initial status |
| `interval` | Training + power monitor | Human/agent analysis | Iteration window metrics, Zeus energy summary |
| `checkpoint` | Training checkpoint hook | Human/agent analysis | Checkpoint path and optional power checkpoint metadata |
| `completed` | Experiment tracker finalize | Human/agent analysis | Final iteration, final status, final power/frequency context |
| `failed` / incomplete finalization | Runtime/log fallback | Human/agent analysis | Error context from logs and partial artifacts |

## Message Queues / Brokers
- None.
- Event persistence happens through `events.jsonl`, `run.json`, and training logs.

## Event Schemas
- `events.jsonl` uses one JSON object per line.
- Important payloads include interval iteration number, Zeus metrics, and checkpoint/finalization metadata.
- For short-run power sweeps, the final Zeus line in the main log is treated as an important derived event even if finalization is incomplete.

## Error Handling
- If `run.json` is incomplete, read `events.jsonl`.
- If `events.jsonl` is insufficient, parse the training log for `[Zeus] Steps ...` summaries and final iteration lines.
- If a detached run exits unexpectedly, inspect the corresponding `_screen_boot` log first.
