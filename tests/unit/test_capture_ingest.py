"""Unit tests for src.dataset.capture.ingest — capture session ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL", reason="Pillow required for ingest tests")

import random  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

from src.dataset.capture.config import (  # noqa: E402
    CaptureConfig,
    ConsentSettings,
    ImageRequirements,
)
from src.dataset.capture.exif import GPS_IFD_TAG, inspect_metadata  # noqa: E402
from src.dataset.capture.ingest import (  # noqa: E402
    SessionMeta,
    ingest_session,
    init_captures_tree,
    is_eval_locked,
    lock_eval_set,
    verify_captures_tree,
)
from src.dataset.manifest import CaptureSessionManifest, SourceManifest  # noqa: E402
from src.utils.dataset_utils import extract_group_key  # noqa: E402


def _make_image(path: Path, size: tuple[int, int] = (640, 480), color: int = 0) -> Path:
    # Seeded 8×8 black/white block pattern → distinct, deterministic aHashes,
    # so the flip-robust perceptual dedup reliably keeps different images apart.
    rng = random.Random(color)  # noqa: S311 — deterministic test pattern, not crypto
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    bw, bh = size[0] // 8, size[1] // 8
    for by in range(8):
        for bx in range(8):
            val = 255 if rng.random() > 0.5 else 0
            draw.rectangle(
                (bx * bw, by * bh, (bx + 1) * bw - 1, (by + 1) * bh - 1), fill=(val, val, val)
            )
    img.save(path)
    return path


def _config(tmp_path: Path, **image_kwargs: object) -> CaptureConfig:
    return CaptureConfig(
        inbox_dir=tmp_path / "inbox",
        captures_root=tmp_path / "captures",
        eval_root=tmp_path / "eval",
        image=ImageRequirements(min_dim=100, **image_kwargs),  # type: ignore[arg-type]
        consent=ConsentSettings(required=False),
    )


def _meta(session_id: str = "h01_kitchen_s001") -> SessionMeta:
    house_id, room = session_id.split("_")[0], session_id.split("_")[1]
    return SessionMeta(
        session_id=session_id,
        house_id=house_id,
        room=room,
        lighting="daylight",
        capture_device="TestCam",
        captured_at="2026-07-20",
        consent_reference="",
        trusted_classes=("gas_cylinder", "stove"),
    )


@pytest.mark.unit
class TestIngestSession:
    """Inbox → validated capture tree."""

    def test_happy_path(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        for i in range(3):
            _make_image(config.inbox_dir / f"IMG_{i}.jpg", color=i * 40)

        result = ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        assert result.accepted == 3
        assert result.rejected == []
        images = sorted((config.captures_root / "images").iterdir())
        assert [p.name for p in images] == [
            "h01_kitchen_s001_0001.jpg",
            "h01_kitchen_s001_0002.jpg",
            "h01_kitchen_s001_0003.jpg",
        ]
        manifest = CaptureSessionManifest.load(
            config.captures_root / "manifests" / "h01_kitchen_s001.json"
        )
        assert manifest.session_id == "h01_kitchen_s001"
        assert manifest.house_id == "h01"
        assert manifest.room == "kitchen"
        assert manifest.image_count == 3
        assert len(manifest.image_hashes) == 3
        assert manifest.trusted_classes == ["gas_cylinder", "stove"]
        assert manifest.annotation_status == "unannotated"

        aggregate = SourceManifest.load(config.captures_root / "manifest.json")
        assert aggregate.source == "custom_captures"
        assert aggregate.image_count == 3
        assert aggregate.query["sessions"] == ["h01_kitchen_s001"]

    def test_rejects_bad_inputs(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "good.jpg")
        _make_image(config.inbox_dir / "small.jpg", size=(80, 80))
        (config.inbox_dir / "broken.jpg").write_bytes(b"not an image at all")
        _make_image(config.inbox_dir / "wrong.gif")

        result = ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        assert result.accepted == 1
        reasons = dict(result.rejected)
        assert "too_small" in reasons["small.jpg"]
        assert "corrupt_image" in reasons["broken.jpg"]
        assert "disallowed_extension" in reasons["wrong.gif"]

    def test_rejects_duplicates_within_session(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        original = _make_image(config.inbox_dir / "a_first.jpg", color=5)
        (config.inbox_dir / "z_copy.jpg").write_bytes(original.read_bytes())

        result = ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        assert result.accepted == 1
        assert len(result.rejected) == 1
        assert "duplicate_of" in result.rejected[0][1]

    def test_append_continues_sequence_and_rejects_reingested(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg", color=1)
        ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        # Second run: same inbox image (already ingested) + one new image.
        _make_image(config.inbox_dir / "two.jpg", color=90)
        result = ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        assert result.accepted == 1
        assert len(result.rejected) == 1  # the re-offered duplicate
        images = sorted((config.captures_root / "images").iterdir())
        assert [p.name for p in images] == [
            "h01_kitchen_s001_0001.jpg",
            "h01_kitchen_s001_0002.jpg",
        ]
        manifest = CaptureSessionManifest.load(
            config.captures_root / "manifests" / "h01_kitchen_s001.json"
        )
        assert manifest.image_count == 2
        assert len(manifest.image_hashes) == 2

    def test_strips_metadata_on_ingest(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        img = Image.new("RGB", (640, 480), (90, 60, 30))
        exif = Image.Exif()
        exif[0x0110] = "PhoneCam"
        gps = exif.get_ifd(GPS_IFD_TAG)
        gps[1] = "N"
        img.save(config.inbox_dir / "phone.jpg", exif=exif)

        ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

        ingested = config.captures_root / "images" / "h01_kitchen_s001_0001.jpg"
        assert inspect_metadata(ingested)["clean"] is True

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg")

        result = ingest_session(
            config.inbox_dir, _meta(), config, config.captures_root, dry_run=True
        )

        assert result.accepted == 1
        assert result.manifest_path is None
        assert not config.captures_root.exists()

    def test_missing_inbox_raises(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        with pytest.raises(FileNotFoundError):
            ingest_session(config.inbox_dir, _meta(), config, config.captures_root)

    def test_locked_eval_refuses_ingest(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg")
        config.eval_root.mkdir(parents=True)
        (config.eval_root / "LOCKED.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="LOCKED"):
            ingest_session(config.inbox_dir, _meta(), config, config.eval_root)


@pytest.mark.unit
class TestEvalLock:
    """Eval set freezing."""

    def test_lock_and_is_locked(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg")
        ingest_session(config.inbox_dir, _meta(), config, config.eval_root)

        assert is_eval_locked(config.eval_root) is False
        lock_path = lock_eval_set(config.eval_root)
        assert is_eval_locked(config.eval_root) is True

        recorded = json.loads(lock_path.read_text(encoding="utf-8"))
        assert recorded["image_count"] == 1
        assert len(recorded["content_digest"]) == 64

    def test_relock_raises(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg")
        ingest_session(config.inbox_dir, _meta(), config, config.eval_root)
        lock_eval_set(config.eval_root)
        with pytest.raises(ValueError, match="already locked"):
            lock_eval_set(config.eval_root)

    def test_lock_empty_set_raises(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        init_captures_tree(config.eval_root)
        with pytest.raises(ValueError, match="empty"):
            lock_eval_set(config.eval_root)


@pytest.mark.unit
class TestVerifyCapturesTree:
    """--verify-all re-validation."""

    def _ingested(self, tmp_path: Path) -> CaptureConfig:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        for i in range(2):
            _make_image(config.inbox_dir / f"img_{i}.jpg", color=i * 70)
        ingest_session(config.inbox_dir, _meta(), config, config.captures_root)
        return config

    def test_clean_tree_verifies(self, tmp_path: Path) -> None:
        config = self._ingested(tmp_path)
        assert verify_captures_tree(config.captures_root, config) == []

    def test_absent_tree_is_ok(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert verify_captures_tree(config.captures_root, config) == []

    def test_detects_tampered_image(self, tmp_path: Path) -> None:
        config = self._ingested(tmp_path)
        target = config.captures_root / "images" / "h01_kitchen_s001_0001.jpg"
        _make_image(target, color=200)  # overwrite with different content
        problems = verify_captures_tree(config.captures_root, config)
        assert any("hash mismatch" in p for p in problems)

    def test_detects_orphan_image(self, tmp_path: Path) -> None:
        config = self._ingested(tmp_path)
        _make_image(config.captures_root / "images" / "h01_kitchen_s001_0099.jpg", color=99)
        problems = verify_captures_tree(config.captures_root, config)
        assert any("orphan image" in p for p in problems)

    def test_detects_orphan_label(self, tmp_path: Path) -> None:
        config = self._ingested(tmp_path)
        (config.captures_root / "labels" / "h01_kitchen_s001_0042.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
        problems = verify_captures_tree(config.captures_root, config)
        assert any("orphan label" in p for p in problems)

    def test_detects_eval_digest_mismatch(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.inbox_dir.mkdir()
        _make_image(config.inbox_dir / "one.jpg")
        ingest_session(config.inbox_dir, _meta(), config, config.eval_root)
        lock_eval_set(config.eval_root)
        # Tamper AFTER locking (bypassing ingest).
        _make_image(config.eval_root / "images" / "h01_kitchen_s001_0001.jpg", color=123)
        problems = verify_captures_tree(config.eval_root, config)
        assert any("LOCKED digest mismatch" in p for p in problems)


@pytest.mark.unit
class TestGroupKeyContract:
    """Merged custom-capture filenames must group by session (split integrity).

    Merge renames images to ``{source}_{original}``; the group-key pattern in
    src/utils/dataset_utils.py must reduce that to the session prefix so all
    images of one session land in the same split.
    """

    def test_session_images_share_group_key(self) -> None:
        assert (
            extract_group_key("custom_captures_h01_kitchen_s001_0007.jpg")
            == "custom_captures_h01_kitchen_s001"
        )
        assert (
            extract_group_key("custom_captures_h01_kitchen_s001_0001.jpg")
            == "custom_captures_h01_kitchen_s001"
        )

    def test_multiword_room_groups_correctly(self) -> None:
        assert (
            extract_group_key("custom_captures_h12_pooja_room_s003_0002.jpg")
            == "custom_captures_h12_pooja_room_s003"
        )

    def test_sessions_do_not_collide(self) -> None:
        key_a = extract_group_key("custom_captures_h01_kitchen_s001_0001.jpg")
        key_b = extract_group_key("custom_captures_h01_kitchen_s002_0001.jpg")
        assert key_a != key_b

    def test_ingested_names_match_session_grammar(self) -> None:
        # The names ingest produces (pre-merge) also group by session.
        assert extract_group_key("h01_kitchen_s001_0007.jpg") == "h01_kitchen_s001"
