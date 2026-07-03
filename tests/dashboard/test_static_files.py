"""Static file content validation — no Node.js testing required.

Verifies security constraints in the client-side assets:
  - No eval() calls (XSS / code-injection vector)
  - No innerHTML assignments using API-sourced data
  - CSS and JS files exist on disk
  - HTML template exists and references the correct asset paths
  - No hardcoded secrets or API keys
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC = Path(__file__).parents[2] / "app" / "dashboard" / "static"
_TEMPLATES = Path(__file__).parents[2] / "app" / "dashboard" / "templates"

_JS = _STATIC / "dashboard.js"
_CSS = _STATIC / "dashboard.css"
_HTML = _TEMPLATES / "dashboard.html"


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestFilesExist:
    def test_js_file_exists(self):
        assert _JS.exists(), "dashboard.js is missing"

    def test_css_file_exists(self):
        assert _CSS.exists(), "dashboard.css is missing"

    def test_html_template_exists(self):
        assert _HTML.exists(), "dashboard.html is missing"


# ---------------------------------------------------------------------------
# JS security — no eval, no unsafe innerHTML
# ---------------------------------------------------------------------------


class TestJSSecurity:
    def setup_method(self):
        self._js = _JS.read_text(encoding="utf-8")

    def test_no_eval_calls(self):
        """eval() is prohibited — it is an XSS / code-injection vector."""
        # Allow 'evaluate' but not a bare eval( call
        bare_eval = re.findall(r"\beval\s*\(", self._js)
        assert bare_eval == [], f"Found eval() calls: {bare_eval}"

    def test_no_inner_html_from_api_data(self):
        """innerHTML must never be set with API-sourced data.

        The codebase should only use textContent or explicit DOM construction
        (createElement / appendChild) when handling data from the server.
        Only the tooltip.textContent = '' clear (which is safe) and
        container.textContent = '' clears are acceptable.
        Assignments to innerHTML that are not empty-string resets are a risk.
        """
        # Find all innerHTML assignments
        assignments = re.findall(r"\.innerHTML\s*=\s*(.+)", self._js)
        for rhs in assignments:
            rhs_stripped = rhs.strip().strip(";").strip("'").strip('"')
            # Only allow empty-string resets
            assert rhs_stripped == "", f"Unsafe innerHTML assignment detected: .innerHTML = {rhs!r}"

    def test_no_document_write(self):
        assert "document.write(" not in self._js

    def test_uses_text_content_for_user_data(self):
        """textContent must be used for inserting data from the API."""
        assert ".textContent" in self._js

    def test_uses_create_element_for_dom(self):
        """createElement must be used for constructing table rows / cards."""
        assert "createElement" in self._js

    def test_uses_strict_mode(self):
        assert "'use strict'" in self._js or '"use strict"' in self._js

    def test_no_hardcoded_api_keys(self):
        js_lower = self._js.lower()
        for keyword in ("api_key", "apikey", "secret", "private_key"):
            assert (
                keyword not in js_lower
            ), f"Potential secret keyword '{keyword}' found in dashboard.js"

    def test_no_external_urls(self):
        """No external fetch/XHR URLs — SVG namespace constants are allowed."""
        # Exclude the W3C SVG namespace URI (a constant identifier, not a network call)
        external = re.findall(r"https?://(?!localhost|127\.0\.0\.1|www\.w3\.org)", self._js)
        assert external == [], f"External URL references found: {external}"


# ---------------------------------------------------------------------------
# CSS validation
# ---------------------------------------------------------------------------


class TestCSSContent:
    def setup_method(self):
        self._css = _CSS.read_text(encoding="utf-8")

    def test_css_has_dark_background(self):
        """Dashboard must use a dark theme (--bg: #0f172a or similar)."""
        assert "--bg:" in self._css

    def test_css_has_card_styles(self):
        assert ".card" in self._css

    def test_css_has_table_styles(self):
        assert ".data-table" in self._css

    def test_css_no_external_imports(self):
        """No @import from external URLs."""
        imports = re.findall(r'@import\s+["\']https?://', self._css)
        assert imports == [], f"External @import found in CSS: {imports}"

    def test_css_has_responsive_breakpoint(self):
        assert "@media" in self._css


# ---------------------------------------------------------------------------
# HTML template validation
# ---------------------------------------------------------------------------


class TestHTMLTemplate:
    def setup_method(self):
        self._html = _HTML.read_text(encoding="utf-8")

    def test_html_references_self_css(self):
        assert "/dashboard/static/dashboard.css" in self._html

    def test_html_references_self_js(self):
        assert "/dashboard/static/dashboard.js" in self._html

    def test_html_no_external_script_src(self):
        """No script tags pointing to external origins."""
        external_scripts = re.findall(r'<script[^>]+src=["\']https?://', self._html)
        assert external_scripts == [], f"External scripts found: {external_scripts}"

    def test_html_no_external_link_href(self):
        """No link tags pointing to external stylesheets."""
        external_links = re.findall(r'<link[^>]+href=["\']https?://', self._html)
        assert external_links == [], f"External links found: {external_links}"

    def test_html_has_lang_attribute(self):
        assert 'lang="' in self._html

    def test_html_has_viewport_meta(self):
        assert 'name="viewport"' in self._html

    def test_html_has_charset_utf8(self):
        assert 'charset="UTF-8"' in self._html or "charset=utf-8" in self._html.lower()

    def test_html_has_paper_warning(self):
        assert "PAPER" in self._html

    def test_html_no_inline_scripts_with_eval(self):
        inline_scripts = re.findall(r"<script[^>]*>(.*?)</script>", self._html, re.DOTALL)
        for script in inline_scripts:
            assert "eval(" not in script, "eval() found in inline script"

    def test_html_no_hardcoded_secrets(self):
        html_lower = self._html.lower()
        for kw in ("api_key", "secret", "private_key", "password"):
            assert kw not in html_lower, f"'{kw}' found in HTML template"

    def test_html_has_aria_live_regions(self):
        assert "aria-live" in self._html

    def test_html_has_role_attributes(self):
        assert 'role="' in self._html

    def test_html_js_at_end_of_body(self):
        """Script tag should appear after the main content (end of body)."""
        js_pos = self._html.find("dashboard.js")
        footer_pos = self._html.find("</footer>")
        assert js_pos > footer_pos, "dashboard.js script should be after </footer>"

    def test_html_has_all_section_ids(self):
        required_ids = [
            "launch-status-badge",
            "error-banner",
            "warnings-bar",
            "card-signal",
            "card-balance",
            "card-equity",
            "card-pnl",
            "card-position",
            "card-trades-count",
            "card-winrate",
            "current-eval",
            "position-container",
            "signal-filter",
            "signals-tbody",
            "trades-empty",
            "trades-table-wrapper",
            "trades-tbody",
            "equity-chart-container",
            "frozen-config-grid",
            "system-status-grid",
        ]
        for elem_id in required_ids:
            assert f'id="{elem_id}"' in self._html, f'Missing element: id="{elem_id}"'
