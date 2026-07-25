from pathlib import Path


FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


def test_product_ui_defaults_to_real_microphone_without_debug_controls() -> None:
    html = FRONTEND.read_text(encoding="utf-8")

    assert "useMock: URL_CONFIG.get('debug') === '1'" in html
    assert ".controls {\n    display: none;" in html
    assert "body.debug-mode .controls { display: flex; }" in html
    assert 'aria-label="Start recording"' in html
    assert "streamWsUrl: `${WS_SCHEME}://${API_AUTHORITY}/streaming/sessions`" in html
    for word in ("BIRD", "FOLLOW", "FORWARD", "HAPPY", "LEARN"):
        assert f"target:'{word}'" in html
    assert "sentence:'A happy bird can fly'" in html
    assert "task:'sentence_pronunciation'" in html
    assert "expected_text: exercise.sentence" in html
    assert "echoCancellation: false" in html
    assert "noiseSuppression: false" in html
    assert "silentFrameCount >= 40" in html
    assert "} else if (data.type === 'pronunciation.evaluated') {" in html
    assert "Voice detected" in html
    assert "Math.sqrt(energy / Math.max(mono.length, 1))" in html
    assert "Different word heard. Try again." in html
    assert "Word recognized — pronunciation check unavailable." in html
    assert "No clear speech heard. Try again." in html
