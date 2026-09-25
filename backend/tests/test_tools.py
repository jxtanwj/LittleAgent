"""Unit tests for the tool functions: pure functions, no model and no network."""

from littleagent.main import get_weather


def test_get_weather_returns_city_in_result() -> None:
    result = get_weather("San Francisco")
    assert "San Francisco" in result


def test_get_weather_is_deterministic() -> None:
    assert get_weather("Beijing") == get_weather("Beijing")


def test_get_weather_handles_distinct_cities() -> None:
    assert get_weather("Shanghai") != get_weather("Beijing")
