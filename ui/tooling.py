import logging


logger = logging.getLogger(__name__)


TOOL_LABELS = {

    "web_search":
        {
            "start":
                "🔎 Searching web...",

            "end":
                "✓ Web search completed",
        },

}



def get_tool_start_message(
    tool_name: str,
) -> str:

    tool = TOOL_LABELS.get(
        tool_name
    )


    if tool:

        return tool["start"]


    return (
        f"⚙️ Running {tool_name}..."
    )



def get_tool_end_message(
    tool_name: str,
) -> str:

    tool = TOOL_LABELS.get(
        tool_name
    )


    if tool:

        return tool["end"]


    return (
        f"✓ {tool_name} completed"
    )
