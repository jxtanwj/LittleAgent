"""The agent kernel: what the agent is made of, independent of how it is reached.

Layering rule, which is the whole point of this package existing: core depends on
nothing above it. Anything may import core - the HTTP layer, a CLI, a TUI - but
core must never import them. Crossing that line is how a kernel turns into a
junk drawer and how the agent starts requiring a web server to be constructed.

Contains:
    tools   the callable tools the agent can use
    agent   assembly of the model, tools and checkpointer, cached for the process
"""
