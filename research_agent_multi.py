#!/usr/bin/env python3
"""Multi-agent research variant: a planner agent splits a topic into subtopics,
independent researcher agents investigate each one in parallel, and a
synthesizer agent merges their findings into one markdown report.

Usage:
    python research_agent_multi.py "impact of microplastics on soil health"
    python research_agent_multi.py "topic" --subtopics 5 --output-dir reports
"""

import argparse
import asyncio
import json
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

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. Given a \
topic, split it into distinct, non-overlapping research angles that together \
give thorough coverage of the topic.

Respond with ONLY a JSON array of short subtopic strings (no other text, no \
markdown fences). Each string should be specific enough that a researcher \
could investigate it independently of the others."""

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant investigating one \
specific angle of a broader topic. Use web search to gather current, \
credible information on your assigned angle only, then write a concise \
research memo in markdown with:

- Inline citations as markdown links where you use a source
- A "Sources" section at the end listing every URL you relied on

Your final message must contain ONLY the memo: no preamble, no commentary \
about your search process. Start directly with a level-2 heading (##) naming \
your angle."""

SYNTHESIZER_SYSTEM_PROMPT = """You are an editor. You will be given a topic \
and several research memos written independently by different researchers, \
each covering a different angle of that topic. Synthesize them into a single \
cohesive markdown report:

- One executive summary covering the whole topic
- Logically organized sections (reorganize across memos where it reads \
better than following memo boundaries)
- Resolve redundancy between memos; note any genuine disagreement between \
sources rather than papering over it
- One deduplicated "Sources" section merging every URL cited across all memos

Your final message must contain ONLY the finished report: no preamble, no \
commentary. Start directly with a level-1 title heading (#)."""


def slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "topic"


async def run_agent(
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    max_turns: int,
    on_search=None,
    label: str = "agent",
) -> tuple[str, float]:
    """Run one Claude agent to completion and return (final_text, cost_usd).

    Degrades gracefully: if the underlying CLI errors out (e.g. it hit
    max_turns before finishing), this returns whatever text was produced in
    earlier turns instead of raising, so one struggling agent in a fan-out
    doesn't take the whole pipeline down.
    """
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )

    last_text = ""
    cost = 0.0

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                text_blocks = [b.text for b in message.content if isinstance(b, TextBlock)]
                if text_blocks:
                    last_text = "\n".join(text_blocks)
                if on_search:
                    for block in message.content:
                        if isinstance(block, ToolUseBlock) and block.name == "WebSearch":
                            on_search(block.input.get("query", ""))
            elif isinstance(message, ResultMessage):
                cost = getattr(message, "total_cost_usd", None) or 0.0
    except Exception as exc:
        print(f"  [{label}] agent error, using partial output: {exc}", file=sys.stderr)

    # Safety net: drop anything before the first heading, in case the model
    # added preamble despite instructions.
    match = re.search(r"^#{1,6} .+", last_text, re.MULTILINE)
    if match:
        last_text = last_text[match.start():]

    return last_text.strip(), cost


def parse_subtopics(raw: str, fallback_topic: str, count: int) -> list[str]:
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        try:
            subtopics = json.loads(raw[start : end + 1])
            subtopics = [str(s).strip() for s in subtopics if str(s).strip()]
            if subtopics:
                return subtopics
        except json.JSONDecodeError:
            pass
    # Fallback: if planning failed to produce parseable JSON, research the
    # topic as a single angle rather than failing the whole run.
    print("  planner output wasn't parseable JSON; falling back to 1 angle", file=sys.stderr)
    return [fallback_topic]


async def plan(topic: str, subtopic_count: int) -> list[str]:
    prompt = f"Topic: {topic}\n\nSplit this into {subtopic_count} research angles."
    raw, _ = await run_agent(
        prompt, PLANNER_SYSTEM_PROMPT, allowed_tools=[], max_turns=2, label="planner"
    )
    return parse_subtopics(raw, topic, subtopic_count)


async def research_subtopic(topic: str, subtopic: str, index: int, max_turns: int) -> tuple[str, float]:
    def on_search(query_str: str) -> None:
        print(f"  [{index}] searching: {query_str}", file=sys.stderr)

    print(f"  [{index}] researching: {subtopic}", file=sys.stderr)
    prompt = (
        f"Broader topic: {topic}\n"
        f"Your specific angle to research: {subtopic}\n\n"
        "Research this angle thoroughly and write the memo."
    )
    memo, cost = await run_agent(
        prompt,
        RESEARCHER_SYSTEM_PROMPT,
        allowed_tools=["WebSearch"],
        max_turns=max_turns,
        on_search=on_search,
        label=f"researcher {index}",
    )
    print(f"  [{index}] done", file=sys.stderr)
    return memo, cost


async def synthesize(topic: str, memos: list[str], max_turns: int) -> tuple[str, float]:
    joined = "\n\n---\n\n".join(memos)
    prompt = f"Topic: {topic}\n\nResearch memos to synthesize:\n\n{joined}"
    return await run_agent(
        prompt, SYNTHESIZER_SYSTEM_PROMPT, allowed_tools=[], max_turns=max_turns, label="synthesizer"
    )


async def multi_agent_research(
    topic: str, subtopic_count: int, subagent_max_turns: int, synth_max_turns: int
) -> str:
    print("Planning subtopics...", file=sys.stderr)
    subtopics = await plan(topic, subtopic_count)
    for i, s in enumerate(subtopics, start=1):
        print(f"  [{i}] {s}", file=sys.stderr)

    print(f"Researching {len(subtopics)} subtopics in parallel...", file=sys.stderr)
    results = await asyncio.gather(
        *(
            research_subtopic(topic, subtopic, i, subagent_max_turns)
            for i, subtopic in enumerate(subtopics, start=1)
        )
    )
    memos = [memo for memo, _ in results if memo]
    research_cost = sum(cost for _, cost in results)

    if not memos:
        return ""

    print("Synthesizing final report...", file=sys.stderr)
    report, synth_cost = await synthesize(topic, memos, synth_max_turns)

    total_cost = research_cost + synth_cost
    print(f"Total cost: ${total_cost:.4f}", file=sys.stderr)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the markdown report into (default: reports)",
    )
    parser.add_argument(
        "--subtopics",
        type=int,
        default=4,
        help="Number of parallel research angles to split the topic into (default: 4)",
    )
    parser.add_argument(
        "--subagent-max-turns",
        type=int,
        default=8,
        help="Max agent turns per researcher subagent (default: 8)",
    )
    parser.add_argument(
        "--synth-max-turns",
        type=int,
        default=6,
        help="Max agent turns for the synthesizer agent (default: 6)",
    )
    args = parser.parse_args()

    print(f"Researching (multi-agent): {args.topic}", file=sys.stderr)
    report = asyncio.run(
        multi_agent_research(
            args.topic, args.subtopics, args.subagent_max_turns, args.synth_max_turns
        )
    )

    if not report:
        print("No report was generated.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"{timestamp}-{slugify(args.topic)}-multi.md"
    output_path.write_text(report)

    print(f"Report written to {output_path}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
