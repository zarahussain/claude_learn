#!/usr/bin/env python3
"""Research agent: give it a topic, it searches the web and writes a markdown report.

Usage:
    python research_agent.py "impact of microplastics on soil health"
    python research_agent.py "topic" --output-dir reports --max-turns 8
"""

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

SYSTEM_PROMPT = """You are a research assistant. Given a topic, use web search to \
gather current, credible information, then produce a well-organized markdown \
report with:

- A short executive summary
- Clearly headed sections covering the key aspects of the topic
- Inline citations as markdown links where you use a source
- A "Sources" section at the end listing every URL you relied on

Prioritize accuracy and cite sources for factual claims. Note any notable \
disagreement between sources rather than papering over it.

Your final message must contain ONLY the markdown report itself: no preamble \
like "Here is the report", no commentary about your search process, and \
nothing after the Sources section. Start the final message directly with the \
report's title heading."""


def slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "topic"


async def research(topic: str, max_turns: int) -> str:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["WebSearch"],
        max_turns=max_turns,
    )

    last_text = ""

    async for message in query(
        prompt=f"Research this topic and write the markdown report: {topic}",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            text_blocks = [b.text for b in message.content if isinstance(b, TextBlock)]
            if text_blocks:
                last_text = "\n".join(text_blocks)
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name == "WebSearch":
                    query_str = block.input.get("query", "")
                    print(f"  searching: {query_str}", file=sys.stderr)
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None)
            if cost is not None:
                print(f"  done (cost: ${cost:.4f})", file=sys.stderr)

    # Safety net: if the model still added preamble before the report despite
    # instructions, drop everything before the first top-level heading.
    match = re.search(r"^# .+", last_text, re.MULTILINE)
    if match:
        last_text = last_text[match.start():]

    return last_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the markdown report into (default: reports)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Max agent turns for searching/writing (default: 6)",
    )
    args = parser.parse_args()

    print(f"Researching: {args.topic}", file=sys.stderr)
    report = asyncio.run(research(args.topic, args.max_turns))

    if not report:
        print("No report was generated.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"{timestamp}-{slugify(args.topic)}.md"
    output_path.write_text(report)

    print(f"Report written to {output_path}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
