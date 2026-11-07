"""Tests for remap_epg's untyped-input boundaries."""

from tools import remap_epg


def test_parse_name_coerces_untyped_values() -> None:
    assert remap_epg.parse_name(None) == ("", [], False)
    assert remap_epg.parse_name(5) == ("", ["5"], False)


def test_parse_id_coerces_untyped_values() -> None:
    assert remap_epg.parse_id(None) == ("", [], False)
    assert remap_epg.parse_id(4647) == ("", ["4647"], False)


def test_query_variants_accepts_untyped_name() -> None:
    assert remap_epg.query_variants(None, "") == [
        {"country": "", "tokens": [], "from_name": True},
    ]
    assert remap_epg.query_variants(4, "") == [
        {"country": "", "tokens": ["4"], "from_name": True},
    ]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
