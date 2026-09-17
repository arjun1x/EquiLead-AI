"""Browser tests: sign-in, live figures, decision form, mobile layout, reduced motion, no-WebGL fallback.
Run: python -m pytest -q -p no:cacheprovider test_browser.py

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
    env = {**os.environ, "DEMO_MODE": "1", "API_PORT": str(port), "PYTHONDONTWRITEBYTECODE": "1", "SESSION_SECRET": "browser-test",
           "RATE_LIMIT_PER_MINUTE": "1000", "DATABASE_URL": f"sqlite:///{(tmp / 'browser.sqlite3').as_posix()}"}
    proc = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(300):
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
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=GL_ARGS)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


def page_for(browser, viewport=None, **kw):
    context = browser.new_context(viewport=viewport or {"width": 1440, "height": 1000}, **kw)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.errors = errors
    return page


def sign_in(page, base, role="reviewer"):
    page.goto(base + "/login")
    page.select_option("#demo-role", role)
    page.click("text=Open demo workspace")
    page.wait_for_url(base + "/")


def test_sign_in_and_dashboard(browser, base):
    page = page_for(browser)
    sign_in(page, base)
    assert page.locator("h1").inner_text().startswith("Your queue")
    assert page.locator(".metric-row article").count() == 4
    assert page.locator("text=Demonstration").first.is_visible()
    assert page.errors == []


def test_live_figures_come_from_the_server_and_submission_scores(browser, base):
    page = page_for(browser)
    sign_in(page, base)
    page.goto(base + "/applications/new")
    page.fill("#f-applicant_name", "Browser Case"); page.click("[data-next]")
    page.fill("#f-annual_income", "120000"); page.fill("#f-verified_monthly_income", "9500"); page.fill("#f-employment_years", "6"); page.fill("#f-monthly_debt", "2400"); page.fill("#f-cash_reserves", "40000"); page.click("[data-next]")
    page.fill("#f-credit_score", "735"); page.fill("#f-credit_history_years", "12"); page.fill("#f-delinquencies_24m", "0"); page.fill("#f-inquiries_6m", "1"); page.click("[data-next]")
    page.fill("#f-property_value", "650000"); page.fill("#f-mortgage_balance", "310000"); page.fill("#f-requested_amount", "90000")
    page.wait_for_function("document.querySelector('[data-live=cltv_after]').textContent.includes('%')")
    assert page.text_content("[data-live=cltv_after]").strip() == "61.5%"
    assert page.text_content("[data-calc-status]").strip() == "from the server"
    page.click("[data-next]")
    for key in ("income_verification", "identity", "property_valuation", "mortgage_statement", "insurance", "credit_authorization"):
        page.check(f"input[value={key}]")
    assert page.locator("#application-review div").count() > 15
    page.click("[data-submit]")
    page.wait_for_url(re.compile(r"/applications/\d+$"))
    assert page.locator(".data-table.compare tbody tr").count() == 3
    assert page.locator("text=Consensus").first.is_visible()
    assert page.errors == []


def test_validation_returns_to_the_right_step(browser, base):
    page = page_for(browser)
    sign_in(page, base)
    page.goto(base + "/applications/new")
    page.fill("#f-applicant_name", "Bad Score")
    page.click("[data-next]"); page.click("[data-next]")            # step 2 is empty -> browser validation stops here
    assert page.locator("[data-step='1']").is_visible() and page.locator("[data-step='2']").is_hidden()


def test_decision_form_reveals_override_reason(browser, base):
    page = page_for(browser)
    sign_in(page, base)
    page.goto(base + "/applications?filter=needs_decision")
    page.click(".data-table tbody tr a.ref >> nth=0")
    page.wait_for_selector("[data-decision-form]")
    recommended = page.get_attribute("[data-decision-form]", "data-recommended")
    other = "declined" if recommended != "declined" else "approved"
    page.check(f"input[name=human_decision][value={other}]")
    assert page.locator("[data-override-field]").is_visible()
    page.check(f"input[name=human_decision][value={recommended}]")
    if recommended != "manual_review":
        assert page.locator("[data-override-field]").is_hidden()
    assert page.errors == []


def test_mobile_layout_has_no_horizontal_overflow(browser, base):
    page = page_for(browser, viewport={"width": 390, "height": 844})
    sign_in(page, base)
    for path in ("/", "/applications", "/applications/1", "/models", "/fairness"):
        page.goto(base + path)
        assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), path
    page.click(".menu-toggle")
    assert page.locator(".sidebar").evaluate("el => el.classList.contains('open')")
    page.keyboard.press("Escape")
    assert page.evaluate("document.activeElement.classList.contains('menu-toggle')")


def test_login_scene_fallback_without_webgl_and_reduced_motion(browser, base):
    page = page_for(browser, reduced_motion="reduce")
    page.add_init_script(NO_WEBGL)
    page.goto(base + "/login")
    page.wait_for_timeout(800)
    assert page.locator(".scene-fallback").is_visible()
    assert page.locator("[data-scene] canvas").count() == 0
    assert page.errors == []
