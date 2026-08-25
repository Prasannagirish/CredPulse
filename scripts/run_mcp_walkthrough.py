"""Drives the CreditPulse MCP server end-to-end via a real MCP client, over stdio, exactly
as spec §8 Phase 4 requires: health check -> drift report -> retrain -> review -> promote
with confirmation. Prints a transcript suitable for pasting into the README. Everything
below is a backtest over historical Lending Club data, not a live production system."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _print_result(label: str, result: types.CallToolResult) -> dict | None:
    print(f"\n--- {label} ---")
    if getattr(result, "isError", False):
        print("(error result)")
    parsed = None
    for block in result.content:
        if isinstance(block, types.TextContent):
            try:
                parsed = json.loads(block.text)
                print(json.dumps(parsed, indent=2))
            except (json.JSONDecodeError, TypeError):
                print(block.text)
    return parsed


async def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(repo_root)
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Connected. Available tools:", [t.name for t in tools.tools])

            health = await session.call_tool("get_model_health", {})
            _print_result("get_model_health()", health)

            drift = await session.call_tool("get_drift_report", {"window": "2019-06"})
            _print_result('get_drift_report(window="2019-06")', drift)

            retrain = await session.call_tool("trigger_retrain", {})
            retrain_data = _print_result("trigger_retrain()", retrain)

            refused = await session.call_tool("promote_challenger", {"confirm": False})
            _print_result("promote_challenger(confirm=False) — expected refusal", refused)

            # Review the comparison before deciding — this is the whole point of the
            # confirm-gate: promotion is a decision, not an automatic consequence of
            # retraining, even in this scripted walkthrough.
            if retrain_data and retrain_data.get("challenger_wins"):
                print(
                    f"\nChallenger (AUC {retrain_data['challenger_auc']:.4f}) beat champion "
                    f"(AUC {retrain_data['champion_auc']:.4f}) — proceeding to promote."
                )
                promoted = await session.call_tool("promote_challenger", {"confirm": True})
                _print_result("promote_challenger(confirm=True)", promoted)
            else:
                print(
                    "\nChallenger did not outperform the champion on this window "
                    f"(challenger AUC {retrain_data.get('challenger_auc') if retrain_data else '?'} vs "
                    f"champion AUC {retrain_data.get('champion_auc') if retrain_data else '?'}) "
                    "— not promoting. This is the correct outcome for a losing challenger, "
                    "not a failure of the walkthrough."
                )


if __name__ == "__main__":
    asyncio.run(main())
