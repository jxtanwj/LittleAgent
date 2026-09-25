from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"{city} is always sunny!"


def main() -> None:
    model = init_chat_model(
        "deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}},
    )
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
    )

    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()
