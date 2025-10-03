import requests
from crewai.tools import tool


# Slack tools require explicit credentials passed per call. No environment fallback.


@tool("SlackListChannelsTool")
def slack_list_channels(token: str, limit: int = 100, cursor: str = None, team_id: str | None = None, channel_ids: str | None = None) -> str:
    """
    List public or specified channels in the workspace with pagination.
    Required:
      - token (str): Slack Bot token. No environment fallback.
    Either:
      - team_id (str) when listing public channels, or
      - channel_ids (comma-separated string) to fetch specific channels by ID.
    Returns:
      JSON response string from Slack API.
    """
    try:
        if not token:
            raise ValueError("SlackListChannelsTool requires 'token' parameter; no environment fallback.")
        

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        predefined_channel_ids = channel_ids
        if not predefined_channel_ids:
            if not team_id:
                raise ValueError("SlackListChannelsTool requires 'team_id' when 'channel_ids' is not provided.")
            params = {
                "types": ["public_channel"],
                "exclude_archived": True,
                "limit": min(limit, 200),
                "team_id": team_id,
            }
            if cursor:
                params["cursor"] = cursor

            response = requests.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.text
        else:
            predefined_channel_ids_array = [cid.strip() for cid in predefined_channel_ids.split(",") if cid.strip()]
            channels = []
            for channel_id in predefined_channel_ids_array:
                params = {"channel": channel_id}
                response = requests.get(
                    "https://slack.com/api/conversations.info",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                if data.get("ok") and data.get("channel") and not data["channel"].get("is_archived"):
                    channels.append(data["channel"])
            result = {
                "ok": True,
                "channels": channels,
                "response_metadata": {"next_cursor": ""}
            }
            return str(result)
    except Exception as e:
        return f"Error listing channels: {str(e)}"


@tool("SlackPostMessageTool")
def slack_post_message(channel_id: str, text: str, token: str) -> str:
    """
    Post a new message to a Slack channel.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - channel_id (str)
      - text (str)
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackPostMessageTool requires 'token' parameter; no environment fallback.")
        
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }


        payload = {"channel": channel_id, "text": text}
        response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error posting message: {str(e)}"


@tool("SlackReplyToThreadTool")
def slack_reply_to_thread(channel_id: str, thread_ts: str, text: str, token: str) -> str:
    """
    Reply to a specific message thread in Slack.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - channel_id (str)
      - thread_ts (str): e.g. '1234567890.123456'
      - text (str)
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackReplyToThreadTool requires 'token' parameter; no environment fallback.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {"channel": channel_id, "thread_ts": thread_ts, "text": text}
        response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error replying to thread: {str(e)}"


@tool("SlackAddReactionTool")
def slack_add_reaction(channel_id: str, timestamp: str, reaction: str, token: str) -> str:
    """
    Add a reaction emoji to a message.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - channel_id (str)
      - timestamp (str)
      - reaction (str) without ::
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackAddReactionTool requires 'token' parameter; no environment fallback.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {"channel": channel_id, "timestamp": timestamp, "name": reaction}
        response = requests.post("https://slack.com/api/reactions.add", headers=headers, json=payload)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error adding reaction: {str(e)}"


@tool("SlackGetChannelHistoryTool")
def slack_get_channel_history(channel_id: str, token: str, limit: int = 10) -> str:
    """
    Get recent messages from a channel.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - channel_id (str)
    Optional:
      - limit: Number of messages (default 10)
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackGetChannelHistoryTool requires 'token' parameter; no environment fallback.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        params = {"channel": channel_id, "limit": limit}
        response = requests.get("https://slack.com/api/conversations.history", headers=headers, params=params)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting channel history: {str(e)}"


@tool("SlackGetThreadRepliesTool")
def slack_get_thread_replies(channel_id: str, thread_ts: str, token: str) -> str:
    """
    Get all replies in a message thread.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - channel_id (str)
      - thread_ts (str)
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackGetThreadRepliesTool requires 'token' parameter; no environment fallback.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        params = {"channel": channel_id, "ts": thread_ts}
        response = requests.get("https://slack.com/api/conversations.replies", headers=headers, params=params)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting thread replies: {str(e)}"


@tool("SlackGetUsersTool")
def slack_get_users(token: str, team_id: str, limit: int = 100, cursor: str = None) -> str:
    """
    Get a list of all users in the workspace with their basic profile information.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - team_id (str)
    Optional:
      - limit (int): Maximum number of users (default 100, max 200)
      - cursor (str): Pagination cursor
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackGetUsersTool requires 'token' parameter; no environment fallback.")
        if not team_id:
            raise ValueError("SlackGetUsersTool requires 'team_id' parameter.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        params = {"limit": min(limit, 200), "team_id": team_id}
        if cursor:
            params["cursor"] = cursor
        response = requests.get("https://slack.com/api/users.list", headers=headers, params=params)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting users: {str(e)}"


@tool("SlackGetUserProfileTool")
def slack_get_user_profile(user_id: str, token: str) -> str:
    """
    Get detailed profile information for a specific user.
    Required:
      - token (str): Slack Bot token. No environment fallback.
      - user_id (str)
    Returns:
      JSON response from Slack API as a string
    """
    try:
        if not token:
            raise ValueError("SlackGetUserProfileTool requires 'token' parameter; no environment fallback.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        params = {"user": user_id, "include_labels": True}
        response = requests.get("https://slack.com/api/users.profile.get", headers=headers, params=params)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting user profile: {str(e)}"