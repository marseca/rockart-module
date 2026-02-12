from pathlib import Path

import cv2

from dstretch_cli.cli import main


def test_end_to_end(tmp_path: Path):
    fixture = Path("tests/fixtures/tiny.png")
    out_path = tmp_path / "out.png"

    code = main([
        "--input",
        str(fixture),
        "--output",
        str(out_path),
        "--scale",
        "2.0",
    ])
    assert code == 0
    assert out_path.exists()

    img = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    assert img is not None
