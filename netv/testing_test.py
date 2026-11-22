"""Tests for testing.py: the fakes behave like the boundaries they replace."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.message import Message
from types import ModuleType
from urllib import error, request

import asyncio
import hashlib
import ipaddress
import json
import os
import pathlib
import signal
import subprocess
import threading
import time

from pychromecast.error import PyChromecastError

import pytest
import uvicorn

from netv import auth, fast_start, testing, util


@pytest.mark.asyncio
async def test_process_ends_on_its_own_with_exit_code():
    process = testing.FakeProcess(exit_code=3, stderr=[b"a\n", b"b\n"])
    assert process.stderr is not None
    assert [await process.stderr.readline() for _ in range(3)] == [b"a\n", b"b\n", b""]
    assert process.returncode is None
    assert await process.wait() == 3
    assert (process.returncode, process.waits, process.signals) == (3, 1, [])


@pytest.mark.asyncio
async def test_stream_read_drains_remaining_lines():
    stream = testing.FakeStream([b"a", b"b"])
    assert await stream.read() == b"ab"
    assert await stream.readline() == b""


@pytest.mark.asyncio
async def test_running_process_waits_until_signalled():
    process = testing.FakeProcess(exit_code=None)
    waiter = asyncio.ensure_future(process.wait())
    await asyncio.sleep(0)
    assert not waiter.done()
    process.terminate()
    assert await waiter == -15
    process.kill()
    assert process.returncode == -15
    assert process.signals == ["TERM", "KILL"]


@pytest.mark.asyncio
async def test_deferred_exit_arrives_on_next_loop_iteration():
    process = testing.FakeProcess(exit_code=None, deferred_exit=True)
    process.terminate()
    assert process.returncode is None
    assert await process.wait() == -15


def test_stubborn_process_needs_kill():
    process = testing.FakeProcess(exit_code=None, ignores_terminate=True)
    process.terminate()
    assert process.returncode is None
    process.kill()
    assert process.returncode == -9


def test_exited_and_reaped_processes():
    assert testing.FakeProcess(exited=True, exit_code=None).returncode == 0
    assert testing.FakeProcess(exited=True, exit_code=2).returncode == 2
    gone = testing.FakeProcess(reaped=True)
    with pytest.raises(ProcessLookupError):
        gone.terminate()
    with pytest.raises(ProcessLookupError):
        gone.kill()


@pytest.mark.asyncio
async def test_launcher_hands_out_queued_then_running_processes():
    queued = testing.FakeProcess(exit_code=1)
    launcher = testing.ProcessLauncher([queued])
    first = await launcher("ffmpeg", "-i", "x", stderr=subprocess.PIPE)
    second = await launcher("ffprobe")
    assert first is queued
    assert second.exit_code is None
    assert launcher.commands == [["ffmpeg", "-i", "x"], ["ffprobe"]]
    assert launcher.processes == [first, second]
    assert launcher.launches[0].options == {"stderr": subprocess.PIPE}


@pytest.mark.asyncio
async def test_launcher_runs_the_programs_effect(tmp_path: pathlib.Path):
    def write_output(command: list[str]) -> None:
        pathlib.Path(command[-1]).write_text("#EXTM3U\n")

    launcher = testing.ProcessLauncher(on_launch=write_output)
    await launcher("ffmpeg", str(tmp_path / "out.m3u8"))
    assert (tmp_path / "out.m3u8").read_text() == "#EXTM3U\n"


@pytest.mark.asyncio
async def test_launcher_awaits_async_effects_and_fails_on_raise():
    gate = asyncio.Event()

    async def blocked(command: list[str]) -> None:
        del command
        await gate.wait()

    launcher = testing.ProcessLauncher(on_launch=blocked)
    pending = asyncio.ensure_future(launcher("ffmpeg"))
    await asyncio.sleep(0)
    assert launcher.launches == []
    gate.set()
    await pending
    assert launcher.commands == [["ffmpeg"]]

    def refuse(command: list[str]) -> None:
        raise RuntimeError(command[0])

    failing = testing.ProcessLauncher(on_launch=refuse)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        await failing("ffmpeg")
    assert failing.launches == []


@pytest.mark.asyncio
async def test_clock_wait_for_times_out_what_is_still_pending():
    clock = testing.FakeClock(now=0)

    async def instant() -> str:
        return "done"

    assert await clock.async_wait_for(instant(), 5) == "done"
    assert await clock.async_wait_for(instant(), None) == "done"
    never = asyncio.Event()
    with pytest.raises(TimeoutError):
        await clock.async_wait_for(never.wait(), 5)
    assert (clock.now, clock.sleeps) == (5, [5])


def test_server_install_without_modules_patches_uvicorn(monkeypatch: pytest.MonkeyPatch):
    recorder = testing.ServerRecorder()
    testing.install_server(monkeypatch, recorder)
    uvicorn.run("netv.main:app", port=1)
    assert recorder.runs == [testing.ServerRun(app="netv.main:app", options={"port": 1})]


@pytest.mark.asyncio
async def test_launcher_error_is_raised():
    launcher = testing.ProcessLauncher(error=FileNotFoundError("ffmpeg"))
    with pytest.raises(FileNotFoundError):
        await launcher("ffmpeg")
    assert launcher.launches == []


def test_run_matches_executable_and_argument():
    run = testing.FakeRun()
    run.program("ffprobe", testing.Program(stdout="{}"))
    run.program("ffprobe", testing.Program(stderr=b"bad", returncode=1), when="broken.ts")
    ok = run(["/usr/bin/ffprobe", "good.ts"], capture_output=True, text=True)
    bad = run(["ffprobe", "broken.ts"], capture_output=True)
    assert (ok.returncode, ok.stdout, ok.stderr) == (0, "{}", "")
    assert (bad.returncode, bad.stdout, bad.stderr) == (1, b"", b"bad")
    assert run.calls == [["/usr/bin/ffprobe", "good.ts"], ["ffprobe", "broken.ts"]]
    assert run.options[0] == {"capture_output": True, "text": True}


def test_run_text_mode_decodes_bytes():
    run = testing.FakeRun()
    run.program("lspci", testing.Program(stdout=b"VGA [8086:1234]"))
    assert run(["lspci"], text=True).stdout == "VGA [8086:1234]"


def test_run_raises_for_errors_and_unknown_programs():
    run = testing.FakeRun()
    run.program("ffmpeg", testing.Program(error=subprocess.TimeoutExpired("ffmpeg", 5)))
    with pytest.raises(subprocess.TimeoutExpired):
        run(["ffmpeg"])
    with pytest.raises(FileNotFoundError):
        run(["missing"])


def test_popen_runs_until_signalled_or_finished():
    popen = testing.PopenRecorder()
    process = popen(["ffmpeg", "-i", "x"], stdout=subprocess.DEVNULL)
    assert popen.started == [process]
    assert process.options == {"stdout": subprocess.DEVNULL}
    assert process.poll() is None
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    process.ignores_terminate = True
    process.terminate()
    assert process.poll() is None
    process.kill()
    assert (process.wait(), process.signals) == (-9, ["TERM", "KILL"])
    finished = popen(["ffmpeg"])
    finished.finish(0)
    finished.terminate()
    assert finished.wait() == 0
    with pytest.raises(subprocess.TimeoutExpired):
        popen(["ffmpeg"]).wait()
    missing = testing.PopenRecorder(error=FileNotFoundError("ffmpeg"))
    with pytest.raises(FileNotFoundError):
        missing(["ffmpeg"])
    assert missing.started == []


def test_response_reads_in_chunks_and_closes():
    response = testing.FakeResponse(b"abcdef", status=201, headers={"Content-Type": "image/png"})
    with response as body:
        assert (body.read(2), body.read(-1), body.read()) == (b"ab", b"cdef", b"")
        assert body.headers.get("content-type") == "image/png"
        assert body.getcode() == 201
    assert response.closed
    assert testing.FakeResponse(b"abc").read1(2) == b"ab"


def test_probed_video_json_reports_a_video_frame_size_without_audio():
    (stream,) = json.loads(testing.probed_video_json())["streams"]
    assert (stream["codec_type"], stream["height"]) == ("video", 1080)


def test_probe_capture_is_the_json_ffprobe_reported():
    report = json.loads(testing.probe_capture("ota_1080i_mpeg2"))
    video = next(s for s in report["streams"] if s["codec_type"] == "video")
    assert (video["codec_name"], video["height"], video["field_order"]) == (
        "mpeg2video",
        1080,
        "tt",
    )
    assert report["format"]["format_name"] == "mpegts"
    with pytest.raises(KeyError):
        testing.probe_capture("missing")


def test_ts_source_opens_with_tables_and_a_replayable_keyframe():
    source = testing.ts_source(12, packets=3)
    assert len(source) == 188 * 6
    assert source[:188] == testing.pat_packet()
    assert source[188:376] == testing.pmt_packet()
    assert fast_start.keyframe_replay(source) == source
    assert fast_start.first_video_pts(source) == 12


def test_only_a_keyframe_packet_is_a_random_access_point():
    keyframe = testing.video_packet(3, keyframe=True)
    plain = testing.video_packet(3)
    assert len(keyframe) == len(plain) == 188
    assert fast_start.first_video_pts(keyframe) == fast_start.first_video_pts(plain) == 3
    tables = testing.pat_packet() + testing.pmt_packet()
    assert fast_start.keyframe_replay(tables + plain * 3) is None
    assert fast_start.keyframe_replay(testing.pat_packet() + keyframe * 3) is None


@pytest.mark.asyncio
async def test_sink_records_writes_and_waits_for_them():
    sink = testing.FakeSink()
    waiter = asyncio.ensure_future(sink.wait_for(3))
    await asyncio.sleep(0)
    sink.write(b"ab")
    await asyncio.sleep(0)
    assert not waiter.done()
    sink.write(b"c")
    await waiter
    await sink.drain()
    sink.close()
    assert (bytes(sink.written), sink.closed) == (b"abc", True)


@pytest.mark.asyncio
async def test_sink_can_fail_writes_or_block_drain():
    failing = testing.FakeSink(error=BrokenPipeError())
    with pytest.raises(BrokenPipeError):
        failing.write(b"x")
    assert failing.written == b""
    drain = asyncio.ensure_future(testing.FakeSink(blocked=True).drain())
    await asyncio.sleep(0)
    assert not drain.done()
    drain.cancel()


def test_upstream_serves_fed_chunks_then_end_or_failure():
    upstream = testing.FakeUpstream(headers={"Content-Type": "video/mp2t"})
    upstream.feed(b"ab")
    upstream.fail(ConnectionResetError())
    upstream.end()
    assert upstream.read1(1) == b"ab"
    with pytest.raises(ConnectionResetError):
        upstream.read1()
    assert upstream.read1() == b""
    assert upstream.headers.get("content-type") == "video/mp2t"


def test_closing_an_upstream_unblocks_a_pending_read():
    upstream = testing.FakeUpstream()
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.append(upstream.read1()))
    reader.start()
    upstream.close()
    reader.join(timeout=5)
    assert (chunks, upstream.closed) == ([b""], True)


def test_ended_upstream_sends_its_data_then_ends():
    default = testing.ended_upstream()
    assert (default.read1(), default.read1()) == (testing.ts_source(), b"")
    assert testing.ended_upstream(b"x").read1() == b"x"


def test_upstreams_open_one_per_request_and_close_them_all():
    upstreams = testing.FakeUpstreams()
    first, second = upstreams(), upstreams()
    assert first is not second
    assert upstreams.opened == [first, second]
    upstreams.close()
    assert all(upstream.closed for upstream in upstreams.opened)
    assert (first.read1(), second.read1()) == (b"", b"")


def test_web_routes_by_url_and_records_requests():
    fresh = 0

    def factory() -> testing.FakeResponse:
        nonlocal fresh
        fresh += 1
        return testing.FakeResponse(b"new")

    web = testing.FakeWeb(
        {
            "http://a/x": testing.FakeResponse(b"ok"),
            "http://a/fail": error.HTTPError("http://a/fail", 503, "down", Message(), None),
            "http://a/fresh": factory,
        },
    )
    req = request.Request("http://a/x", headers={"Accept": "*/*"})
    assert web(req, timeout=3).read() == b"ok"
    assert web("http://a/fresh").geturl() == "http://a/fresh"
    assert fresh == 1
    with pytest.raises(error.HTTPError):
        web("http://a/fail")
    with pytest.raises(error.URLError):
        web("http://a/unknown")
    assert web.urls == ["http://a/x", "http://a/fresh", "http://a/fail", "http://a/unknown"]
    assert web.fetches[0].headers == {"Accept": "*/*"}
    assert web.fetches[0].timeout == 3
    assert web.fetches[1].headers == {}


def test_dials_only_the_internet_reads_a_fetch_s_pinned_handlers(monkeypatch: pytest.MonkeyPatch):
    web = testing.FakeWeb({"http://x/": testing.FakeResponse(b"")})
    testing.install_web(monkeypatch, web, util)
    util.safe_urlopen("http://x/").close()
    util.safe_urlopen("http://x/", allow_address=lambda ip: ip.is_global).close()
    util.safe_urlopen("http://x/", allow_address=lambda ip: True).close()
    assert [testing.dials_only_the_internet(f) for f in web.fetches] == [False, True, False]
    # The HTTP and HTTPS handlers each carry the rule; an unpinned fetch has none.
    assert [len(testing.address_vets(f)) for f in web.fetches] == [0, 2, 2]
    [vet, _] = testing.address_vets(web.fetches[2])
    assert vet(ipaddress.ip_address("192.168.1.87"))


def test_played_url_reads_a_player_page_that_hides_it():
    auth.create_user("ann", "pw")
    url = "http://up/movie/u/p/9.mkv"
    ref = auth.stream_ref("ann", url)
    page = f'streamRef: "{ref}",\ndirectUrl: "/direct/{ref}/stream.mkv",'
    assert testing.played_url(page) == url
    # A page that shows the URL, or whose two refs disagree, fails.
    with pytest.raises(ValueError, match="shows the upstream URL"):
        testing.played_url(page + url)
    other = auth.stream_ref("ann", "http://up/other.mkv")
    with pytest.raises(ValueError, match="different streams"):
        testing.played_url(page.replace(f"/direct/{ref}/", f"/direct/{other}/"))
    with pytest.raises(ValueError, match="Not a player page"):
        testing.played_url("no player here")


def test_web_serves_safe_urlopen_with_its_checks(monkeypatch: pytest.MonkeyPatch):
    web = testing.FakeWeb({"https://p.example/a": testing.FakeResponse(b"ok")})
    testing.install_web(monkeypatch, web, util)
    with util.safe_urlopen("https://p.example/a", timeout=7, user_agent="netv") as response:
        assert response.read() == b"ok"
    [fetch] = web.fetches
    assert (fetch.url, fetch.timeout, fetch.headers) == (
        "https://p.example/a",
        7,
        {"User-agent": "netv"},
    )
    assert [type(h).__name__ for h in fetch.handlers] == ["_SafeRedirectHandler"]
    with pytest.raises(error.URLError, match="Unsafe URL scheme"):
        util.safe_urlopen("file:///etc/passwd")
    assert len(web.fetches) == 1


def test_subprocess_installer_replaces_only_given_calls(monkeypatch: pytest.MonkeyPatch):
    module = _module_using_stdlib()
    run, popen = testing.FakeRun(), testing.PopenRecorder()
    run.program("ffmpeg", testing.Program(stdout="v"))
    testing.install_subprocess(monkeypatch, module, run=run)
    assert module.subprocess.Popen is subprocess.Popen
    testing.install_subprocess(monkeypatch, module, popen=popen)
    assert module.subprocess.run(["ffmpeg"], text=True).stdout == "v"
    module.subprocess.Popen(["ffmpeg"])
    assert len(popen.started) == 1
    assert subprocess.run is not run


def test_threads_run_inline_and_keep_errors():
    recorder = testing.ThreadRecorder()
    seen: list[int] = []

    def boom() -> None:
        raise RuntimeError("boom")

    thread = recorder.make_thread(target=seen.append, args=(1,), daemon=True)
    thread.start()
    recorder.make_thread(target=boom, name="b").start()
    recorder.make_thread().start()
    assert seen == [1]
    assert thread.daemon and thread.name == "fake-thread"
    assert not thread.is_alive()
    thread.join(timeout=1)
    thread.run()
    assert seen == [1]
    assert [type(e) for e in recorder.errors] == [RuntimeError]


def test_deferred_threads_run_on_demand_including_children():
    recorder = testing.ThreadRecorder(run_inline=False)
    seen: list[str] = []

    def record(*, name: str) -> None:
        seen.append(name)

    def parent() -> None:
        seen.append("parent")
        recorder.make_thread(target=record, kwargs={"name": "child"}).start()

    thread = recorder.make_thread(target=parent)
    thread.start()
    assert thread.is_alive() and seen == []
    recorder.run_all()
    assert seen == ["parent", "child"]


def test_contended_lock_runs_other_threads_work_once():
    seen: list[str] = []
    lock = testing.ContendedLock(lambda: seen.append("other"))
    with lock as held:
        assert held is True
        seen.append("mine")
    with lock:
        pass
    assert (seen, lock.acquired) == (["other", "mine"], 2)


@pytest.mark.asyncio
async def test_background_tasks_run_or_discard():
    tasks = testing.BackgroundTasks()
    seen: list[str] = []

    async def child() -> None:
        seen.append("child")

    async def parent() -> None:
        seen.append("parent")
        tasks(child())

    first = tasks(parent())
    assert len(tasks.pending) == 1
    await tasks.run_all()
    assert seen == ["parent", "child"]
    assert first.done() and not first.cancelled()
    assert first.cancel() is False
    await first.run()
    held = tasks(child())
    tasks.discard()
    assert (held.cancelled(), held.done(), tasks.pending) == (True, True, [])
    assert seen == ["parent", "child"]


@pytest.mark.asyncio
async def test_clock_moves_only_when_told():
    clock = testing.FakeClock(now=100)
    clock.sleep(2)
    assert await clock.async_sleep(3, result="r") == "r"
    clock.advance(1)
    assert (clock.time(), clock.monotonic(), clock.sleeps) == (106, 106, [2, 3])


def _module_using_stdlib() -> ModuleType:
    module = ModuleType("fake_netv_module")
    module.__dict__.update(
        time=time,
        asyncio=asyncio,
        threading=threading,
        subprocess=subprocess,
    )
    return module


@pytest.mark.asyncio
async def test_installers_replace_only_the_modules_globals(monkeypatch: pytest.MonkeyPatch):
    module = _module_using_stdlib()
    bare = ModuleType("no_stdlib")
    clock = testing.FakeClock(now=5)
    launcher = testing.ProcessLauncher()
    recorder = testing.ThreadRecorder(run_inline=False)
    testing.install_launcher(monkeypatch, launcher, module)
    testing.install_clock(monkeypatch, clock, module, bare)
    testing.install_threads(monkeypatch, recorder, module)

    assert module.time.monotonic() == 5
    assert module.time.perf_counter is time.perf_counter
    await module.asyncio.sleep(10)
    assert module.time.time() == 15
    module.time.sleep(5)
    assert (module.time.time(), clock.sleeps) == (20, [10, 5])
    with pytest.raises(TimeoutError):
        await module.asyncio.wait_for(asyncio.Event().wait(), 3)
    assert module.time.time() == 23
    await module.asyncio.create_subprocess_exec("ffmpeg")
    assert launcher.commands == [["ffmpeg"]]
    assert module.asyncio.Event is asyncio.Event
    module.threading.Thread(target=print).start()
    assert len(recorder.threads) == 1
    assert time.monotonic() != 15
    assert not hasattr(bare, "time")
    monkeypatch.undo()
    assert module.asyncio is asyncio
    assert module.time is time
    assert module.threading is threading


def test_server_recorder_replaces_uvicorn_run(monkeypatch: pytest.MonkeyPatch):
    module = ModuleType("server_module")
    module.__dict__["uvicorn"] = object()
    recorder = testing.ServerRecorder()
    testing.install_server(monkeypatch, recorder, module)
    app = object()
    module.uvicorn.run(app, port=8000)
    assert recorder.runs == [testing.ServerRun(app=app, options={"port": 8000})]


async def _waits(*, sleep: Callable[[float], Awaitable[object]] = asyncio.sleep) -> None:
    await sleep(10)


class _Paced:
    def __init__(self, *, sleep: Callable[[float], object] = time.sleep, limit: int = 3) -> None:
        sleep(limit)


async def _own_sleep(seconds: float) -> None:
    del seconds


async def _other(*, sleep: Callable[[float], Awaitable[object]] = _own_sleep) -> None:
    await sleep(1)


@pytest.mark.asyncio
async def test_install_clock_routes_injected_sleeps_to_the_clock(monkeypatch: pytest.MonkeyPatch):
    module = _module_using_stdlib()
    module.__dict__.update(waits=_waits, Paced=_Paced, other=_other)
    clock = testing.FakeClock(now=0)
    testing.install_clock(monkeypatch, clock, module)
    await _waits()
    _Paced()
    assert clock.sleeps == [10, 3]
    # A default that is not a real sleep stays.
    assert _other.__kwdefaults__ == {"sleep": _own_sleep}
    monkeypatch.undo()
    assert _waits.__kwdefaults__ == {"sleep": asyncio.sleep}
    assert _Paced.__init__.__kwdefaults__ == {"sleep": time.sleep, "limit": 3}


def test_clock_event_waits_pass_on_the_clock(monkeypatch: pytest.MonkeyPatch):
    clock = testing.FakeClock(now=0)
    module = _module_using_stdlib()
    testing.install_clock(monkeypatch, clock, module)
    event = module.threading.Event()
    assert event.wait(5) is False
    assert clock.now == 5
    event.set()
    assert (event.is_set(), event.wait(5), clock.now) == (True, True, 5)
    event.clear()
    with pytest.raises(RuntimeError, match="forever"):
        event.wait()


def test_clock_deadline_ends_wait_loops():
    clock = testing.FakeClock(now=0, deadline=125)
    event = clock.make_event()
    ticks = 0
    while not event.wait(60):
        ticks += 1
    assert (ticks, clock.now, event.is_set()) == (2, 125, True)
    late = testing.FakeClock(now=200, deadline=125).make_event()
    assert late.wait() is True
    assert clock.events == [event]


def test_clock_datetime_and_installed_datetime(monkeypatch: pytest.MonkeyPatch):
    clock = testing.FakeClock(now=0)
    module = _module_using_stdlib()
    module.__dict__["datetime"] = datetime
    testing.install_clock(monkeypatch, clock, module)
    assert module.datetime.now(timezone.utc) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    clock.advance(60)
    assert module.datetime.now(timezone.utc).minute == 1
    assert isinstance(module.datetime.now(), datetime)
    assert datetime.now(timezone.utc).year > 1970


def test_signals_are_recorded_and_can_fail(monkeypatch: pytest.MonkeyPatch):
    module = ModuleType("signal_module")
    module.__dict__["os"] = os
    recorder = testing.SignalRecorder({7: ProcessLookupError("gone")})
    testing.install_signals(monkeypatch, recorder, module)
    module.os.kill(5, signal.SIGTERM)
    with pytest.raises(ProcessLookupError):
        module.os.kill(7, signal.SIGKILL)
    assert recorder.kills == [(5, signal.SIGTERM), (7, signal.SIGKILL)]
    assert module.os.getpid() == os.getpid()


def test_receiver_plays_fails_or_hangs_on_load():
    tv = testing.FakeChromecast(friendly_name="Den")
    controller = tv.media_controller
    controller.play_media("http://n/a.m3u8", "application/x-mpegurl", title="A")
    assert controller.status == testing.FakeMediaStatus(
        content_id="http://n/a.m3u8",
        player_state="PLAYING",
    )
    assert controller.loaded == [
        testing.CastLoad(
            url="http://n/a.m3u8",
            content_type="application/x-mpegurl",
            options={"title": "A"},
        ),
    ]
    controller.pause()
    assert controller.status.player_state == "PAUSED"
    controller.play()
    assert controller.status.player_state == "PLAYING"
    tv.set_volume(0.3)
    assert tv.status == testing.FakeReceiverStatus(volume_level=0.3)
    controller.stop()
    assert controller.status == testing.FakeMediaStatus(player_state="IDLE")
    assert controller.commands == ["load", "pause", "play", "volume", "stop"]
    tv.wait(timeout=1)
    tv.disconnect(timeout=0)
    assert tv.disconnected and tv.cast_info.friendly_name == "Den"
    assert tv.socket_client.is_connected

    failing = testing.FakeMediaController(outcome="fails")
    failing.play_media("u", "t")
    assert (failing.status.content_id, failing.status.idle_reason) == ("u", "ERROR")
    hanging = testing.FakeMediaController(outcome="hangs")
    hanging.play_media("u", "t")
    assert hanging.status == testing.FakeMediaStatus()


def test_receiver_takes_load_secs_to_start():
    clock = testing.FakeClock(now=0)
    tv = testing.FakeChromecast(clock=clock, load_secs=0.2, started_state="BUFFERING")
    tv.media_controller.play_media("u", "t")
    assert tv.media_controller.status.player_state == "IDLE"
    clock.advance(0.1)
    assert tv.media_controller.status.player_state == "IDLE"
    clock.advance(0.1)
    assert tv.media_controller.status == testing.FakeMediaStatus(
        content_id="u",
        player_state="BUFFERING",
    )


def test_thread_join_records_timeout():
    thread = testing.ThreadRecorder().make_thread()
    thread.join(timeout=6)
    thread.join()
    assert thread.joins == [6, None]


def test_receiver_errors_and_missing_status():
    tv = testing.FakeChromecast(error=OSError("gone"))
    with pytest.raises(OSError, match="gone"):
        tv.media_controller.play()
    quiet = testing.FakeChromecast()
    quiet.status = None
    quiet.set_volume(0.5)
    assert quiet.media_controller.commands == ["volume"]


def test_cast_network_connects_and_discovers(monkeypatch: pytest.MonkeyPatch):
    tv = testing.FakeChromecast()
    network = testing.CastNetwork(
        {"192.168.1.5": tv},
        advertised=[testing.FakeCastDevice(host="192.168.1.5", friendly_name="TV")],
    )
    module = ModuleType("cast_module")
    module.__dict__.update(get_chromecast_from_host=None, CastBrowser=None, Zeroconf=None)
    testing.install_chromecast(monkeypatch, network, module)
    assert module.get_chromecast_from_host(("192.168.1.5", 8009), tries=1) is tv
    with pytest.raises(PyChromecastError, match=r"no receiver at 192\.168\.1\.9"):
        module.get_chromecast_from_host(("192.168.1.9", 8009))
    assert network.connects == ["192.168.1.5", "192.168.1.9"]
    assert network.browsing is False
    with module.Zeroconf() as zconf:
        browser = module.CastBrowser(object(), zconf)
        browser.start_discovery()
        assert network.browsing
        assert list(browser.devices.values()) == network.advertised
        browser.stop_discovery()
    assert not network.browsing


def test_cast_network_discovery_error():
    network = testing.CastNetwork(discovery_error=OSError("no multicast"))
    with pytest.raises(OSError, match="multicast"):
        network.zeroconf()


def test_cheap_kdf_is_one_iteration_pbkdf2(monkeypatch: pytest.MonkeyPatch):
    module = ModuleType("kdf_module")
    module.__dict__["hashlib"] = hashlib
    testing.install_cheap_kdf(monkeypatch, module)
    expected = hashlib.pbkdf2_hmac("sha256", b"pw", b"salt", 1)
    assert module.hashlib.pbkdf2_hmac("sha256", b"pw", b"salt", 100_000) == expected
    assert module.hashlib.sha256 is hashlib.sha256


def test_xtream_client_serves_server_info():
    client = testing.FakeXtreamClient(server_info={"server_info": {"timezone": "UTC"}})
    assert client.get_server_info() == {"server_info": {"timezone": "UTC"}}
    assert client.calls == ["get_server_info"]


def test_thread_pool_runs_work_on_threads():
    with testing.thread_pool(max_workers=2, mp_context=None) as pool:
        assert pool.submit(threading.current_thread).result() is not threading.current_thread()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
