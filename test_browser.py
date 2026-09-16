"""Browser tests: desktop, tablet, mobile, reduced motion and no-WebGL. Run: python -m pytest -q test_browser.py

Skips cleanly when the playwright package or a Chromium build is not installed
(python -m pip install playwright && python -m playwright install chromium)."""
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
ROOT = pathlib.Path(__file__).resolve().parent
GL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
NO_WEBGL = ("const g=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(t,...a){"
            "return (t==='webgl'||t==='webgl2'||t==='experimental-webgl')?null:g.call(this,t,...a)};")


@pytest.fixture(scope="module")
def base():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="equilead-browser-"))
    env = {**os.environ, "DEMO_MODE": "1", "API_PORT": str(port), "PYTHONDONTWRITEBYTECODE": "1",
           "RATE_LIMIT_PER_MINUTE": "1000", "DATABASE_URL": f"sqlite:///{(tmp / 'browser.sqlite3').as_posix()}"}
    proc = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.2)
    else:
        proc.terminate(); pytest.skip("server did not start")
    yield url
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch(headless=True, args=GL_ARGS)
        except Exception as exc:  # no browser build installed
            pytest.skip(f"Chromium unavailable: {str(exc).splitlines()[0]}")
        yield b
        b.close()


def demo(browser, base, role="analyst", **context):
    ctx = browser.new_context(**context)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/login")
    page.select_option("#demo-role", role)
    page.click("text=Explore demo workspace")
    page.wait_for_url(base + "/")
    return ctx, page, errors


def test_desktop_scene_form_and_results(browser, base):
    ctx, page, errors = demo(browser, base, viewport={"width": 1440, "height": 900})
    page.wait_for_timeout(800)
    assert page.locator(".scene-canvas canvas").count() == 1
    assert page.locator(".scene-fallback").first.is_hidden()
    page.focus(".scene-canvas"); page.keyboard.press("ArrowRight")
    page.click("[data-scene-pause]")
    assert page.locator("[data-scene-pause]").first.get_attribute("aria-pressed") == "true"
    page.goto(base + "/apply")
    page.fill("#loan", "50000"); page.fill("#value", "400000"); page.fill("#mortdue", "200000")
    assert page.locator("[data-summary=ltv]").inner_text() == "62.5%"
    assert page.locator("[data-summary=equity]").inner_text() == "$150,000"
    page.fill("#loan", ""); page.click("[data-next]")
    assert page.locator(".step-caption").inner_text() == "Step 1 of 3"  # invalid step blocks progress
    page.fill("#loan", "50000"); page.click("[data-next]")
    assert page.locator(".step-caption").inner_text() == "Step 2 of 3"
    assert page.evaluate("document.activeElement.tagName") == "H2"       # focus moves to the step heading
    page.click("[data-next]")
    assert page.locator("#application-review dt").count() == 13
    for outcome, heading in (("approved", "Recommend approval."), ("denied", "Recommend decline."), ("review", "Recommend approval.")):
        page.goto(base + "/apply"); page.click("[data-goto-step='2']")
        page.select_option("#demo_outcome", outcome); page.click("[data-submit]")
        page.wait_for_url(re.compile(r"/decisions/\d+"))
        assert page.locator(".result-banner h2").inner_text() == heading
        assert page.locator(".demo-alert").count() == 1
        assert (page.locator(".draft-warning").count() == 1) == (outcome == "review")
        assert (page.locator(".reason-code").count() > 0) == (outcome == "denied")
    page.goto(base + "/audit?q=jamie&status=review")
    assert page.locator(".records-table tbody tr").count() == 1
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert errors == []
    ctx.close()


def test_reviewer_records_decision_and_sees_monitoring(browser, base):
    ctx, page, errors = demo(browser, base, role="reviewer", viewport={"width": 1440, "height": 900})
    page.goto(base + "/decisions/1")
    page.select_option("#human_decision", "approved"); page.fill("#note", "Verified income documents.")
    page.click("text=Record decision")
    page.wait_for_url(re.compile(r"/decisions/1$"))
    assert "Recorded by a reviewer." in page.locator(".human-panel h2").inner_text()
    page.click("text=Monitoring")
    page.wait_for_url(base + "/monitoring")
    assert page.locator(".monitor-grid .panel").count() >= 3
    assert errors == []
    ctx.close()


def test_tablet_layout(browser, base):
    ctx, page, errors = demo(browser, base, viewport={"width": 820, "height": 1180})
    for path in ("/", "/apply", "/audit", "/decisions/1", "/guide"):
        page.goto(base + path)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), path
    assert page.locator(".menu-toggle").is_hidden()
    assert errors == []
    ctx.close()


def test_mobile_navigation_and_no_horizontal_scroll(browser, base):
    ctx, page, errors = demo(browser, base, viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    for path in ("/", "/apply", "/audit", "/decisions/1", "/guide"):
        page.goto(base + path)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), path
    page.goto(base + "/")
    assert page.locator(".menu-toggle").is_visible()
    page.click(".menu-toggle")
    assert page.locator(".menu-toggle").get_attribute("aria-expanded") == "true"
    assert page.evaluate("document.activeElement.closest('.sidebar') !== null")
    page.keyboard.press("Escape")
    assert page.locator(".menu-toggle").get_attribute("aria-expanded") == "false"
    assert page.evaluate("document.activeElement.classList.contains('menu-toggle')")
    assert errors == []
    ctx.close()


def test_reduced_motion_pauses_scene(browser, base):
    ctx, page, errors = demo(browser, base, viewport={"width": 1200, "height": 800}, reduced_motion="reduce")
    page.wait_for_timeout(600)
    pause = page.locator("[data-scene-pause]").first
    assert pause.get_attribute("aria-pressed") == "true"
    assert pause.get_attribute("aria-label") == "Resume 3D motion"
    assert errors == []
    ctx.close()


def test_no_webgl_shows_svg_fallback(browser, base):
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    ctx.add_init_script(NO_WEBGL)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/login")
    page.wait_for_timeout(600)
    assert page.locator(".scene-canvas canvas").count() == 0
    assert page.locator(".scene-fallback").first.is_visible()
    assert page.locator(".scene-controls").first.is_hidden()
    assert "architectural illustration" in page.locator(".scene-instruction").first.inner_text()
    page.click("text=Explore demo workspace")
    page.wait_for_url(base + "/")  # navigation keeps working without WebGL
    assert errors == []
    ctx.close()


def test_login_keyboard_and_password_toggle(browser, base):
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    page = ctx.new_page()
    page.goto(base + "/login")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.classList.contains('skip-link')")
    page.click("[data-password-toggle]")
    assert page.locator("#password").get_attribute("type") == "text"
    assert page.locator("[data-password-toggle]").get_attribute("aria-pressed") == "true"
    ctx.close()
