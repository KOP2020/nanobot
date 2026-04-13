# Digital Human Demo

This repository is now a minimal digital human demo backend.

## What remains

- `app/`: FastAPI service and digital-human runtime
- `tests/app/`: focused tests for the demo flow
- `pyproject.toml`: simplified project metadata

## Main API

- `POST /api/digital-humans`: create a digital human from `open_sid`
- `POST /api/conversations`: create a conversation for one end user
- `POST /api/conversations/{conversation_id}/messages`: run one chat turn
- `GET /api/tasks/{task_id}`: inspect background paper-reading tasks
- `GET /api/health`: health check

## Run

```bash
python -m pip install -e .
digital-human-demo
```

If `DIGITAL_HUMAN_API_KEY` and `DIGITAL_HUMAN_API_BASE` are not configured, the app starts with a demo echo provider so the service still boots locally.
