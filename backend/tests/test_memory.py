"""Tests for the long-term memory store and the tools built on it.

Storage and retrieval are tested together because neither half is useful alone:
a memory that was saved but cannot be found is indistinguishable from one that
was never written.

Everything here runs against a temporary file (the autouse `memory_file` fixture
in conftest), so the developer's real memories are never read or written.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from littleagent.core import memory, tools
# The agent's tools, which happen to share names with the store's functions.
# The store is always reached through `memory.` so the two never get confused.
from littleagent.core.tools import delete_memory, save_memory, search_memory, update_memory


def write_raw(entries: list[memory.Memory]) -> None:
    """Write the file by hand, the way a hand-edited memories.json looks."""
    memory.MEMORY_FILE.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def temp_files() -> list[str]:
    """Leftover temp files would pile up in the data directory after every crash."""
    return [path.name for path in memory.MEMORY_FILE.parent.iterdir() if path.suffix == ".tmp"]


def tool_config(user_id: str | None = None) -> RunnableConfig:
    """The run config the agent hands to a tool, as the API layer builds it."""
    configurable: dict[str, Any] = {} if user_id is None else {"user_id": user_id}
    return {"configurable": configurable}


def score_of(mem: memory.Memory, query: str) -> float:
    """Score one memory the way the store does, against a corpus of just itself."""
    query_terms = memory._extract_terms(query)
    idf = memory._idf_map([memory._extract_terms(str(mem.get("content", "")))], query_terms)
    return memory._calc_score(mem, query_terms, idf, sum(idf.values()))


# --- loading -----------------------------------------------------------------


def test_load_memories_returns_empty_when_no_file_exists() -> None:
    """A fresh install has no data file at all; that means "no memories", not a crash."""
    assert memory.load_memories() == []


def test_load_memories_returns_empty_for_an_empty_file() -> None:
    """json.loads("") raises, so an empty file must be tolerated rather than fatal."""
    memory.MEMORY_FILE.write_text("", encoding="utf-8")

    assert memory.load_memories() == []


def test_load_memories_returns_empty_for_a_truncated_file() -> None:
    """A crash mid-write can leave broken JSON; the next read must survive it."""
    memory.MEMORY_FILE.write_text('[{"id": 1, "content": "half', encoding="utf-8")

    assert memory.load_memories() == []


def test_load_memories_returns_empty_when_the_json_is_an_object() -> None:
    """A hand-edited file may hold a single object instead of an array."""
    memory.MEMORY_FILE.write_text('{"id": 1, "content": "one"}', encoding="utf-8")

    assert memory.load_memories() == []


def test_load_memories_drops_entries_that_are_not_objects() -> None:
    """A hand-edited [1, 2] used to crash every .get downstream, taking the whole
    memory feature down with it."""
    memory.MEMORY_FILE.write_text('[1, {"id": 2, "content": "ok"}, "x"]', encoding="utf-8")

    assert [mem.get("content") for mem in memory.load_memories()] == ["ok"]
    assert memory.save_memory({"content": "new"}) == 3


def test_reading_a_damaged_file_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Silent degradation is what made the old data loss invisible."""
    memory.MEMORY_FILE.write_text("{ not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert memory.load_memories() == []

    assert caplog.records, "a file that cannot be read must not fail silently"


def test_load_memories_reads_back_what_was_saved() -> None:
    """Unknown fields survive a round trip; the store is not allowed to eat them."""
    memory.save_memory({"content": "用户喜欢喝茶", "note": "extra fields survive"})

    stored = memory.load_memories()

    assert len(stored) == 1
    assert stored[0]["content"] == "用户喜欢喝茶"
    assert stored[0]["note"] == "extra fields survive"


# --- saving ------------------------------------------------------------------


def test_save_memory_assigns_ids_starting_from_one() -> None:
    assert memory.save_memory({"content": "first"}) == 1
    assert memory.save_memory({"content": "second"}) == 2


def test_save_memory_creates_the_data_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing creates the data directory beforehand, so the first save has to."""
    monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "nested" / "memories.json")

    assert memory.save_memory({"content": "first"}) == 1


def test_save_memory_stamps_created_at() -> None:
    memory.save_memory({"content": "first"})

    # Parsing is the assertion: a stamp nothing can parse is as useless as none.
    assert datetime.fromisoformat(memory.load_memories()[0]["created_at"])


def test_save_memory_stamps_the_scope() -> None:
    memory.save_memory({"content": "first"}, scope="alice")

    assert memory.load_memories()[0]["scope"] == "alice"


def test_save_memory_leaves_the_callers_dict_alone() -> None:
    """Stamping id onto the caller's dict is a side effect nobody asked for."""
    payload = {"content": "first"}

    memory.save_memory(payload)

    assert payload == {"content": "first"}


def test_save_memory_uses_the_highest_id_not_the_last_entry() -> None:
    """Hand-edited files can be out of order; reusing an id would merge two memories."""
    write_raw([{"id": 7, "content": "high"}, {"id": 2, "content": "low"}])

    assert memory.save_memory({"content": "new"}) == 8


def test_save_memory_ignores_entries_without_an_integer_id() -> None:
    """Entries with a missing or non-integer id are skipped, not crashed on."""
    write_raw(
        [{"content": "no id"}, {"id": "3", "content": "string id"}, {"id": 2, "content": "ok"}]
    )

    assert memory.save_memory({"content": "new"}) == 3


def test_the_file_stays_human_readable() -> None:
    """ensure_ascii=False is deliberate: the file is meant to be editable by hand."""
    memory.save_memory({"content": "用户喜欢喝茶"})

    assert "用户喜欢喝茶" in memory.MEMORY_FILE.read_text(encoding="utf-8")


def test_saving_the_same_content_twice_keeps_one_memory() -> None:
    """Repeating a fact is normal - the model may re-confirm it every turn -
    and duplicates would eat the retrieval slots and fill the context with copies."""
    first = memory.save_memory(
        {"content": "The user likes tea", "importance": 0.2, "memory_type": "user"}
    )
    second = memory.save_memory(
        {"content": "The user likes tea", "importance": 0.9, "memory_type": "fact"}
    )

    assert first == second
    stored = memory.load_memories()
    assert len(stored) == 1
    assert stored[0]["importance"] == 0.9, "the stronger importance wins"
    assert stored[0]["memory_type"] == "user", "the original type is not rewritten"
    assert stored[0]["created_at"], "the original timestamp is kept"


def test_the_same_content_in_another_scope_is_a_different_memory() -> None:
    memory.save_memory({"content": "The user likes tea"}, scope="alice")
    memory.save_memory({"content": "The user likes tea"}, scope="bob")

    assert len(memory.load_memories()) == 2


# --- atomic writes and backup ------------------------------------------------


def test_saving_leaves_no_temp_files_behind() -> None:
    memory.save_memory({"content": "first"})

    assert temp_files() == []


def test_a_failed_dump_leaves_the_file_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    memory.save_memory({"content": "first"})
    before = memory.MEMORY_FILE.read_text(encoding="utf-8")

    def explode(*args: object, **kwargs: object) -> str:
        raise TypeError("not serialisable")

    monkeypatch.setattr(memory.json, "dumps", explode)

    with pytest.raises(TypeError):
        memory.save_memory({"content": "second"})

    assert memory.MEMORY_FILE.read_text(encoding="utf-8") == before
    assert temp_files() == []


def test_a_failed_replace_leaves_the_file_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader holding the file open (Windows) or a full disk must not lose data."""
    memory.save_memory({"content": "first"})
    before = memory.MEMORY_FILE.read_text(encoding="utf-8")

    def explode(*args: object, **kwargs: object) -> None:
        raise PermissionError("locked by another process")

    monkeypatch.setattr(memory.os, "replace", explode)

    with pytest.raises(PermissionError):
        memory.save_memory({"content": "second"})

    assert memory.MEMORY_FILE.read_text(encoding="utf-8") == before
    assert temp_files() == [], "the temporary file must be cleaned up on failure"


def test_a_corrupt_file_is_backed_up_before_it_is_overwritten() -> None:
    """This is the fix for the worst loss: the tolerant read returned "no memories",
    and the next save wrote that emptiness over everything that was in the file."""
    corrupt = '[{"id": 1, "content": "用户的生日是 3 月 1 日"}, {"id": 2,'
    memory.MEMORY_FILE.write_text(corrupt, encoding="utf-8")

    memory.save_memory({"content": "一条新记忆"})

    backup = memory.MEMORY_FILE.with_name(memory.MEMORY_FILE.name + ".bak")
    assert backup.read_text(encoding="utf-8") == corrupt
    assert [mem["content"] for mem in memory.load_memories()] == ["一条新记忆"]


def test_a_file_that_is_not_an_array_is_backed_up_too() -> None:
    """It parses, but the store still cannot use it, so overwriting would lose it."""
    memory.MEMORY_FILE.write_text('{"id": 1, "content": "hand written"}', encoding="utf-8")

    memory.save_memory({"content": "new"})

    backup = memory.MEMORY_FILE.with_name(memory.MEMORY_FILE.name + ".bak")
    assert "hand written" in backup.read_text(encoding="utf-8")


def test_a_valid_file_is_not_backed_up() -> None:
    memory.save_memory({"content": "first"})

    memory.save_memory({"content": "second"})

    backup = memory.MEMORY_FILE.with_name(memory.MEMORY_FILE.name + ".bak")
    assert not backup.exists()


def test_a_failed_backup_aborts_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backup exists so that data is never overwritten without a copy; if the
    copy cannot be made, overwriting anyway would defeat the whole point."""
    memory.MEMORY_FILE.write_text("{ broken", encoding="utf-8")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(memory.shutil, "copy2", explode)

    with pytest.raises(OSError):
        memory.save_memory({"content": "new"})

    assert memory.MEMORY_FILE.read_text(encoding="utf-8") == "{ broken"


def test_concurrent_saves_all_survive() -> None:
    """Every read-modify-write runs under one lock.

    Without it this loses almost everything: 1200 saves across four threads left
    11 memories behind, with no exception raised anywhere - a read that landed on
    a half-written file reported "no memories", and the next save wrote that over
    the rest.
    """
    per_thread = 50

    def worker(tag: str) -> None:
        for index in range(per_thread):
            memory.save_memory({"content": f"{tag}-{index}"})

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in "abcd"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stored = memory.load_memories()
    assert len(stored) == 4 * per_thread
    ids = [mem["id"] for mem in stored]
    assert len(set(ids)) == len(ids), "ids must stay unique under concurrency"


# --- updating and deleting ---------------------------------------------------


def test_update_memory_merges_fields_and_keeps_the_rest() -> None:
    memory_id = memory.save_memory({"content": "user likes tea", "importance": 0.5})

    assert memory.update_memory(memory_id, {"importance": 0.9}) is True

    stored = memory.get_memory_by_id(memory_id)
    assert stored is not None
    assert stored["importance"] == 0.9
    assert stored["content"] == "user likes tea"


def test_update_memory_stamps_updated_at() -> None:
    memory_id = memory.save_memory({"content": "user likes tea"})

    memory.update_memory(memory_id, {"importance": 0.9})

    stored = memory.get_memory_by_id(memory_id)
    assert stored is not None
    assert datetime.fromisoformat(stored["updated_at"])


def test_update_never_rewrites_the_bookkeeping_fields() -> None:
    """A model echoing a whole memory back must not be able to rewrite its id,
    which would break every reference to it, or move it into someone else's scope."""
    memory_id = memory.save_memory({"content": "user likes tea"})

    updated = memory.update_memory(
        memory_id,
        {
            "id": 99,
            "created_at": "1999-01-01T00:00:00+00:00",
            "scope": "bob",
            "importance": 0.9,
        },
    )

    assert updated is True
    stored = memory.get_memory_by_id(memory_id)
    assert stored is not None
    assert stored["id"] == memory_id
    assert stored["scope"] == memory.DEFAULT_SCOPE
    assert stored["created_at"] != "1999-01-01T00:00:00+00:00"
    assert stored["importance"] == 0.9, "the fields that are allowed still apply"


def test_update_memory_returns_false_for_an_unknown_id() -> None:
    """False rather than an exception: an id may come from the model or a stale listing."""
    assert memory.update_memory(999, {"content": "nothing to update"}) is False


def test_delete_memory_removes_only_the_target() -> None:
    first = memory.save_memory({"content": "first"})
    second = memory.save_memory({"content": "second"})

    assert memory.delete_memory(first) is True

    assert [mem["id"] for mem in memory.load_memories()] == [second]


def test_delete_memory_returns_false_for_an_unknown_id() -> None:
    assert memory.delete_memory(999) is False


def test_get_memory_by_id_returns_none_for_an_unknown_id() -> None:
    assert memory.get_memory_by_id(999) is None


def test_get_memory_by_id_returns_a_snapshot_not_a_handle() -> None:
    """Editing the returned dict must not quietly change what is stored."""
    memory_id = memory.save_memory({"content": "original"})

    found = memory.get_memory_by_id(memory_id)
    assert found is not None
    found["content"] = "edited in memory"

    stored = memory.get_memory_by_id(memory_id)
    assert stored is not None
    assert stored["content"] == "original"


# --- scope -------------------------------------------------------------------


def test_memories_are_invisible_across_scopes() -> None:
    memory.save_memory({"content": "alice likes tea"}, scope="alice")
    memory.save_memory({"content": "bob likes coffee"}, scope="bob")

    found = memory.get_relevant_memories("likes", scope="alice")

    assert [mem["content"] for mem in found] == ["alice likes tea"]


def test_entries_without_a_scope_belong_to_the_global_scope() -> None:
    """Files written before the field existed must not become invisible."""
    write_raw([{"id": 1, "content": "user likes tea", "importance": 0.5}])

    assert len(memory.get_relevant_memories("tea", scope=memory.DEFAULT_SCOPE)) == 1
    assert len(memory.get_relevant_memories("tea")) == 1


def test_scope_none_searches_every_scope() -> None:
    memory.save_memory({"content": "alice likes tea"}, scope="alice")
    memory.save_memory({"content": "bob likes tea"}, scope="bob")

    assert len(memory.get_relevant_memories("tea", scope=None)) == 2


def test_update_and_delete_stay_inside_their_scope() -> None:
    memory_id = memory.save_memory({"content": "alice likes tea"}, scope="alice")

    assert memory.update_memory(memory_id, {"content": "changed"}, scope="bob") is False
    assert memory.delete_memory(memory_id, scope="bob") is False
    assert memory.get_memory_by_id(memory_id, scope="bob") is None
    assert memory.get_memory_by_id(memory_id, scope="alice") is not None


# --- retrieval ---------------------------------------------------------------


def test_no_stored_memories_means_no_matches() -> None:
    assert memory.get_relevant_memories("anything") == []


def test_a_memory_sharing_no_term_with_the_query_is_dropped() -> None:
    """An empty result means "nothing relevant", not "nothing stored"."""
    memory.save_memory({"content": "the user drives a truck"})

    assert memory.get_relevant_memories("coffee") == []


def test_a_query_without_any_terms_matches_nothing() -> None:
    memory.save_memory({"content": "user likes tea"})

    assert memory.get_relevant_memories("??? !!!") == []


def test_a_single_chinese_character_recalls_a_longer_memory() -> None:
    """The reason both sides collect single characters as well as bigrams.

    A bigram-only tokeniser would score "茶" against "用户喜欢喝茶" as no overlap
    at all, and the memory would be invisible to a one-character query.
    """
    memory.save_memory({"content": "用户喜欢喝茶"})

    found = memory.get_relevant_memories("茶")

    assert [mem["content"] for mem in found] == ["用户喜欢喝茶"]


def test_search_is_case_insensitive() -> None:
    memory.save_memory({"content": "The user likes Coffee"})

    assert len(memory.get_relevant_memories("COFFEE")) == 1


def test_word_forms_match_the_same_word_in_any_shape() -> None:
    """Both sides run the same term extractor, so one extra form on either side
    is enough for them to meet."""
    memory.save_memory({"content": "The user likes tea"})
    memory.save_memory({"content": "The user has two classes"})
    memory.save_memory({"content": "The user loved the film"})

    assert len(memory.get_relevant_memories("like")) == 1
    assert len(memory.get_relevant_memories("class")) == 1
    assert len(memory.get_relevant_memories("love")) == 1


def test_the_best_match_comes_first() -> None:
    memory.save_memory({"content": "user drinks coffee"})
    memory.save_memory({"content": "user likes coffee"})

    found = memory.get_relevant_memories("user likes coffee")

    assert [mem["content"] for mem in found] == ["user likes coffee", "user drinks coffee"]


def test_a_rare_term_outweighs_a_common_one() -> None:
    """"user" is in every memory and must not drag an unrelated one into the
    results; "tea" is what the query is about, so it decides the ranking."""
    for index in range(5):
        memory.save_memory({"content": f"user note number {index} about coffee"})
    memory.save_memory({"content": "the user drinks tea"})

    found = memory.get_relevant_memories("user tea", min_score=0.5)

    assert [mem["content"] for mem in found] == ["the user drinks tea"]


def test_importance_breaks_a_tie_at_equal_overlap() -> None:
    memory.save_memory({"content": "user likes tea", "importance": 0.1})
    memory.save_memory({"content": "user likes coffee", "importance": 0.9})

    found = memory.get_relevant_memories("likes")

    assert [mem["id"] for mem in found] == [2, 1]


def test_ties_break_newest_first() -> None:
    """File order puts the oldest first, which is exactly backwards: a stale fact
    would outrank the one that replaced it."""
    memory.save_memory({"content": "user likes tea"})
    newest = memory.save_memory({"content": "user likes coffee"})

    found = memory.get_relevant_memories("likes")

    assert found[0]["id"] == newest


def test_duplicate_contents_collapse_to_the_newest() -> None:
    """Old files may already contain duplicates from before saving deduplicated."""
    write_raw(
        [
            {"id": 1, "content": "user likes tea", "created_at": "2020-01-01T00:00:00+00:00"},
            {"id": 2, "content": "user likes tea", "created_at": "2026-01-01T00:00:00+00:00"},
        ]
    )

    found = memory.get_relevant_memories("tea")

    assert [mem["id"] for mem in found] == [2]


def test_a_single_stored_memory_is_still_findable() -> None:
    """With one document every term appears in every document; without idf
    smoothing the rarity of a term would be zero and the only memory in the
    store could never be retrieved."""
    memory.save_memory({"content": "user likes tea"})

    assert len(memory.get_relevant_memories("tea")) == 1


def test_min_score_drops_weak_matches() -> None:
    """Matching only on a common word is not relevance, and the caller can say so."""
    memory.save_memory({"content": "the user drinks tea"})
    memory.save_memory({"content": "the user drives a truck"})

    assert len(memory.get_relevant_memories("user tea")) == 2
    assert len(memory.get_relevant_memories("user tea", min_score=0.5)) == 1


def test_memory_type_filters_the_candidates() -> None:
    memory.save_memory({"content": "user likes tea", "memory_type": "preference"})
    memory.save_memory({"content": "user was born in 1990", "memory_type": "fact"})

    found = memory.get_relevant_memories("user", memory_type="fact")

    assert [mem["content"] for mem in found] == ["user was born in 1990"]


def test_limit_caps_the_number_of_results() -> None:
    for index in range(6):
        memory.save_memory({"content": f"note {index} about coffee"})

    assert len(memory.get_relevant_memories("coffee", limit=2)) == 2


def test_the_default_limit_is_five() -> None:
    """Six matches and no explicit limit: the tool-facing default is part of the contract."""
    for index in range(6):
        memory.save_memory({"content": f"note {index} about coffee"})

    assert len(memory.get_relevant_memories("coffee")) == 5


def test_a_non_positive_limit_returns_nothing() -> None:
    """A negative limit used to slice from the tail and return the worst matches."""
    memory.save_memory({"content": "user likes tea"})

    assert memory.get_relevant_memories("tea", limit=0) == []
    assert memory.get_relevant_memories("tea", limit=-1) == []


# --- scoring internals -------------------------------------------------------


def test_extract_terms_mixes_languages() -> None:
    terms = memory._extract_terms("The user 喜欢 tea")

    assert {"the", "user", "tea"} <= terms, "english runs are lowercased into whole words"
    assert {"喜", "欢", "喜欢"} <= terms, "chinese yields both single characters and bigrams"


def test_extract_terms_keeps_both_the_word_and_its_stripped_forms() -> None:
    """Keeping only one canonical stem loses pairs where just one side strips."""
    terms = memory._extract_terms("likes classes loved")

    assert {"likes", "like"} <= terms
    assert {"classes", "class", "classe"} <= terms, "es and s are stripped independently"
    assert {"loved", "lov", "love"} <= terms, "the e comes back: loved must meet love"
    assert {"running", "runn", "run"} <= memory._extract_terms("running")


def test_extract_terms_leaves_short_words_alone() -> None:
    """Stripping a short word leaves noise, and "is" would become "i"."""
    assert memory._extract_terms("is") == {"is"}
    assert memory._extract_terms("bus") == {"bus"}


def test_calc_score_gives_importance_at_most_a_quarter_of_the_weight() -> None:
    """The band is [0.75, 1.0] of the lexical score, so a keyword match still dominates."""
    assert score_of({"content": "tea", "importance": 0.0}, "tea") == pytest.approx(0.75)
    assert score_of({"content": "tea", "importance": 1.0}, "tea") == pytest.approx(1.0)


def test_calc_score_clamps_importance_to_its_band() -> None:
    """A model that sends 5.0 must not buy a five-fold bonus."""
    assert score_of({"content": "tea", "importance": 5.0}, "tea") == pytest.approx(1.0)
    assert score_of({"content": "tea", "importance": -1.0}, "tea") == pytest.approx(0.75)


def test_calc_score_defaults_a_missing_or_non_numeric_importance() -> None:
    """Old entries and model output both arrive without a usable importance."""
    assert score_of({"content": "tea"}, "tea") == pytest.approx(0.875)
    assert score_of({"content": "tea", "importance": "high"}, "tea") == pytest.approx(0.875)


def test_calc_score_is_zero_without_a_shared_term() -> None:
    assert score_of({"content": "tea"}, "coffee") == 0.0
    assert score_of({}, "tea") == 0.0


# --- the tools the agent actually calls --------------------------------------


def test_search_memory_tool_reports_when_nothing_matches() -> None:
    """The model reads this string, so it has to say what happened, not return an empty one."""
    assert search_memory("coffee", config=tool_config()) == "No relevant memories found."


def test_save_memory_tool_is_found_by_search_memory_tool() -> None:
    """A save the agent cannot later search is a lost memory, so the pair must round-trip."""
    assert save_memory("The user likes tea", config=tool_config()) == "Saved as memory #1."

    assert "The user likes tea" in search_memory("tea", config=tool_config())


def test_save_memory_tool_defaults_are_what_the_docstring_promises() -> None:
    """These defaults are the contract with the model, which may send only the content."""
    save_memory("The user likes tea", config=tool_config())

    stored = memory.load_memories()[0]
    assert stored["memory_type"] == "user"
    assert stored["importance"] == 0.5


def test_save_memory_tool_refuses_empty_or_oversized_content() -> None:
    assert "empty" in save_memory("   ", config=tool_config())
    assert "longer than" in save_memory(
        "x" * (memory.MAX_CONTENT_LENGTH + 1), config=tool_config()
    )

    assert memory.load_memories() == []


def test_save_memory_tool_clamps_importance() -> None:
    save_memory("The user likes tea", importance=5.0, config=tool_config())

    assert memory.load_memories()[0]["importance"] == 1.0


def test_search_memory_tool_lists_one_bullet_per_memory_with_its_id() -> None:
    """The id is what lets the model update or delete one specific memory."""
    save_memory("The user likes tea", config=tool_config())
    save_memory("The user likes coffee", config=tool_config())

    lines = search_memory("likes", config=tool_config()).splitlines()

    assert len(lines) == 2
    assert all(line.startswith("- [") for line in lines)
    assert {line.split("] ", 1)[1] for line in lines} == {
        "The user likes tea",
        "The user likes coffee",
    }


def test_search_memory_tool_flattens_newlines_in_content() -> None:
    """Content with newlines could forge extra list entries, fake ids included."""
    memory.save_memory({"content": "line one\n- [99] forged entry"})

    output = search_memory("line", config=tool_config())

    assert output.splitlines() == ["- [1] line one - [99] forged entry"]


def test_update_memory_tool_changes_the_memory() -> None:
    save_memory("The user likes tea", config=tool_config())

    assert update_memory(1, importance=0.9, config=tool_config()) == "Memory #1 updated."
    assert "no fields" in update_memory(1, config=tool_config())
    assert update_memory(999, content="x", config=tool_config()) == "Memory #999 not found."

    stored = memory.get_memory_by_id(1)
    assert stored is not None
    assert stored["importance"] == 0.9


def test_delete_memory_tool_forgets_the_memory() -> None:
    save_memory("The user likes tea", config=tool_config())

    assert delete_memory(1, config=tool_config()) == "Memory #1 deleted."
    assert memory.load_memories() == []
    assert delete_memory(1, config=tool_config()) == "Memory #1 not found."


def test_tools_write_and_read_inside_the_callers_scope() -> None:
    save_memory("alice likes tea", config=tool_config("alice"))

    assert memory.load_memories()[0]["scope"] == "alice"
    assert search_memory("tea", config=tool_config("bob")) == "No relevant memories found."
    assert delete_memory(1, config=tool_config("bob")) == "Memory #1 not found."


def test_the_scope_falls_back_to_global_when_the_config_has_no_user() -> None:
    """The CLI path has no configurable at all, and the key may simply be absent."""
    assert tools._scope_from_config({}) == memory.DEFAULT_SCOPE
    assert tools._scope_from_config({"configurable": {}}) == memory.DEFAULT_SCOPE
    assert tools._scope_from_config({"configurable": {"user_id": ""}}) == memory.DEFAULT_SCOPE
    assert tools._scope_from_config({"configurable": {"user_id": "alice"}}) == "alice"


def test_the_model_never_sees_the_config_parameter() -> None:
    """Injection needs the exact annotation. Written as `RunnableConfig | None` it
    stops being injected and shows up as a parameter the model is asked to fill;
    the tool then dies on its first call. Only the agent-path test catches that,
    because every test here hands the config over itself."""
    schema = StructuredTool.from_function(search_memory).tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {"query", "limit", "memory_type"}
