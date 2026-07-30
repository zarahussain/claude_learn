# claude_learn

## Research agent

`research_agent.py` is a small CLI tool built on the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/python).
Give it a topic and it uses Claude's web search tool to research it, then
writes a structured, cited markdown report.

### Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Usage

```bash
python research_agent.py "impact of microplastics on soil health"

# optional flags
python research_agent.py "topic" --output-dir reports --max-turns 8
```

The report is printed to stdout and saved under `reports/` (or `--output-dir`)
as a timestamped markdown file.
