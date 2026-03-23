# ollarena

Terminal-based model comparison arena for Ollama. Compare output quality across your local models — side-by-side, with ELO ratings.

You already benchmark **speed** with `modelbench` and **tool-calling** with `toolprobe`. ollarena benchmarks **quality**.

## Install

```bash
pip install -e .
```

## Usage

### Battle — compare models head-to-head

```bash
# Interactive model selection
ollarena battle "Explain monads in simple terms"

# Specific models
ollarena battle "Write a haiku about recursion" -m llama3.3:latest,mistral:latest

# Blind mode — model names hidden until after voting
ollarena battle "What causes aurora borealis?" --blind

# With system prompt
ollarena battle "Review this code" -s "You are a senior engineer" -m qwen2.5:latest,deepseek-r1:latest

# Pipe from stdin
echo "Summarize this text..." | ollarena battle -

# Limit output length
ollarena battle "Write a short poem" -t 200
```

Models run in parallel. Results displayed side-by-side with token counts and speed stats. Vote on the winner (or tie/skip) — ELO ratings update automatically.

### Leaderboard — track quality over time

```bash
ollarena leaderboard
# or
ollarena lb
```

Shows ELO ratings, win/loss/tie records, and total battles per model.

### History — review past battles

```bash
ollarena history
ollarena history -n 50
```

### List models

```bash
ollarena list
```

Shows available Ollama models with their current ELO rating.

### Reset

```bash
ollarena reset --confirm
```

## Blind Mode

Use `--blind` to hide model names during comparison. You'll see "Contestant A" and "Contestant B" instead of model names. Names are revealed after you vote. This eliminates bias from model reputation.

## How ELO Works

- All models start at 1500
- Winners gain points, losers lose points (K=32)
- Ties split the difference
- More battles = more accurate ratings
- Ratings persist across sessions in `~/.ollarena/`

## Data

Ratings and history are stored in `~/.ollarena/`:
- `ratings.json` — ELO ratings per model
- `history.json` — full battle log

## Requirements

- Python 3.11+
- Ollama running locally
- At least 2 models pulled
