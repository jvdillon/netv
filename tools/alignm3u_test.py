"""Tests for alignm3u."""

import pathlib
import runpy
import sys

import pytest

from tools import alignm3u


def test_parse_callsign() -> None:
    assert alignm3u.parse_callsign("KQEDDT2") == ("KQED", "DT", 2)
    assert alignm3u.parse_callsign("kntvd") == ("KNTV", "D", 1)
    assert alignm3u.parse_callsign("KQSLD") == ("KQSL", "LD", 1)
    assert alignm3u.parse_callsign("12345") == ("12345", "", 1)


def test_parse_m3u(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "in.m3u"
    p.write_text(
        "#EXTM3U\n"
        "orphan-line\n"
        '#EXTINF:-1 channel-number="2.1" tvg-name="KTVU",FOX 2\n'
        "http://hdhr/auto/v2.1\n"
        "\n"
        '#EXTINF:0 tvg-id="x" junk,No Url\n',
    )
    assert alignm3u.parse_m3u(p) == [
        ["FOX 2", {"channel-number": "2.1", "tvg-name": "KTVU"}, "http://hdhr/auto/v2.1"],
        ["No Url", {"tvg-id": "x"}],
    ]


def test_parse_m3u_ignores_directives_between_extinf_and_url(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "in.m3u"
    p.write_text(
        "#EXTM3U\n"
        '#EXTINF:-1 channel-number="2.1",A\n'
        "#EXTVLCOPT:http-user-agent=x\n"
        "http://h/2.1\n"
        "http://h/stray\n",
    )
    assert alignm3u.parse_m3u(p) == [["A", {"channel-number": "2.1"}, "http://h/2.1"]]


def test_main_survives_vlc_options(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m3u = tmp_path / "in.m3u"
    m3u.write_text(
        '#EXTM3U\n#EXTINF:-1 channel-number="2.1",A\n#EXTVLCOPT:http-user-agent=x\nhttp://h/2.1\n',
    )
    xml = tmp_path / "x.xml"
    xml.write_text('<tv><channel id="I2"><display-name>2.1</display-name></channel></tv>')
    out = tmp_path / "o.m3u"
    argv = ["alignm3u.py", "-i", str(m3u), "-x", str(xml), "-o", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert alignm3u.main() == 0
    assert out.read_text().splitlines()[1:] == [
        '#EXTINF:-1 channel-number="2.1" tvg-id="I2" group-title="OTA",A',
        "http://h/2.1",
    ]


def test_align_atsc3_keeps_two_digit_subchannel() -> None:
    lookup = {"2.1": {"wrong"}, "2.10": {"right"}, "15": {"whole"}}
    entries: list[list] = [
        ["A", {"channel-number": "102.10"}, "u"],
        ["B", {"channel-number": "115"}, "u"],
    ]
    alignm3u.align_channels(entries, lookup)
    assert entries[0][1]["tvg-id"] == "right"
    assert entries[1][1]["tvg-id"] == "whole"


def test_parse_m3u_rejects_missing_header(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "bad.m3u"
    p.write_text("#EXTINF:-1,X\n")
    with pytest.raises(ValueError, match="Invalid M3U file"):
        alignm3u.parse_m3u(p)


def test_parse_xmltv_channels_and_lookup(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "g.xml"
    p.write_text(
        "<tv>"
        '<channel id="a"><display-name>2.1</display-name><display-name>KTVU</display-name>'
        "<display-name></display-name><icon/></channel>"
        "<channel><display-name>noid</display-name></channel>"
        '<channel id="b"><display-name>KTVU</display-name></channel>'
        '<programme channel="a"/>'
        "</tv>",
    )
    chans = alignm3u.parse_xmltv_channels(p)
    assert chans == {"a": ("2.1", "KTVU"), "b": ("KTVU",)}
    assert dict(alignm3u.build_lookup(chans)) == {"2.1": {"a"}, "KTVU": {"a", "b"}}


def test_align_channels_priority() -> None:
    lookup = {"2.1": {"num"}, "N": {"name"}, "5.1": {"x", "y"}, "M": {"m1", "m2"}}
    entries: list[list] = [
        ["short", {}],
        ["A", {"channel-number": "2.1", "tvg-name": "N"}, "u"],
        ["B", {"channel-number": "9.9", "tvg-name": "N"}, "u"],
        ["C", {"channel-number": "5.1", "tvg-name": "M"}, "u"],
        ["M", {}, "u"],
        ["D", {"channel-number": "102.1"}, "u"],
        ["E", {"channel-number": "7.1"}, "u"],
    ]
    aligned, missing = alignm3u.align_channels(entries, lookup)
    assert aligned is entries
    assert entries[0] == ["short", {}]
    assert entries[1][1]["tvg-id"] == "num"
    assert entries[2][1]["tvg-id"] == "name"
    assert entries[3][1]["tvg-id"] in {"x", "y"}
    assert entries[4][1]["tvg-id"] in {"m1", "m2"}
    assert entries[5][1]["tvg-id"] == "num"  # ATSC3 102.1 -> 2.1.
    assert missing == [entries[6]]


def test_write_m3u(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "o.m3u"
    entries = [
        ["short", {}],
        ["T", {"tvg-name": "Named", "group-title": "News", "channel-id": "105"}, "u1"],
        ["Plain", {"channel-id": "5"}, "u2"],
    ]
    alignm3u.write_m3u(entries, out, "http://srv/x.xml", "LOCAL")
    assert out.read_text() == (
        '#EXTM3U url-tvg="http://srv/x.xml" x-tvg-url="http://srv/x.xml"\n'
        '#EXTINF:-1 tvg-name="Named" group-title="LOCAL | News | ATSC3" channel-id="105",Named\n'
        "u1\n"
        '#EXTINF:-1 channel-id="5" group-title="LOCAL",Plain\n'
        "u2\n"
    )
    alignm3u.write_m3u([["X", {}, "u"]], out, "")
    assert out.read_text() == '#EXTM3U\n#EXTINF:-1 group-title="OTA",X\nu\n'


def _inputs(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    m3u = tmp_path / "lineup.m3u"
    m3u.write_text(
        "#EXTM3U\n"
        '#EXTINF:-1 channel-number="2.1" tvg-name="KTVU",KTVU\nhttp://h/2.1\n'
        '#EXTINF:-1 channel-number="9.1",KQED\nhttp://h/9.1\n',
    )
    xml = tmp_path / "x.xml"
    xml.write_text('<tv><channel id="I2"><display-name>2.1</display-name></channel></tv>')
    return m3u, xml, tmp_path / "out.m3u"


def test_main_cli(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    m3u, xml, out = _inputs(tmp_path)
    argv = ["alignm3u.py", "-i", str(m3u), "-x", str(xml), "-o", str(out), "--group", "G"]
    monkeypatch.setattr(sys, "argv", argv)
    assert alignm3u.main() == 0
    text = capsys.readouterr().out
    assert "  Found 2 channels\n" in text
    assert "  Found 1 channels\n" in text
    assert "Unable to align 1 channels:\n  9.1: KQED" in text
    assert "Aligned 1/2 channels" in text
    assert 'tvg-id="I2"' in out.read_text()
    assert 'group-title="G"' in out.read_text()


def test_script_entry_point(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    m3u, xml, out = _inputs(tmp_path)
    xml.write_text(
        '<tv><channel id="I2"><display-name>2.1</display-name></channel>'
        '<channel id="I9"><display-name>KQED</display-name></channel></tv>',
    )
    argv = ["alignm3u.py", "--input", str(m3u), "--xmltv", str(xml), "--output", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--xmltv-url", "http://s/x.xml"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(alignm3u.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert "Unable to align" not in capsys.readouterr().out
    assert out.read_text().startswith('#EXTM3U url-tvg="http://s/x.xml"')


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
