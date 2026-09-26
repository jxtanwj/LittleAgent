"""Tools the agent can call.

A plain function is enough: create_agent reads the name, the docstring and the
type annotations to build the schema it shows the model, so the docstring is part
of the tool's contract with the model rather than a comment for humans.

This is the natural place for a plugin system to hook in later - a registry that
tool modules register with - but there is no registry yet because one plugin
would not justify the indirection.
"""

from __future__ import annotations


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"{city} is always sunny!"
