"""
Test Configuration & Fixtures
================================
Shared pytest fixtures for the entire test suite.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_label_content():
    """A valid YOLO label file content (class_id cx cy w h)."""
    return "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.25\n"


@pytest.fixture
def temp_dataset(tmp_path, sample_label_content):
    """
    Create a temporary dataset directory structure
    mimicking the YOLO format for testing.
    """
    # Create directories
    (tmp_path / "images" / "all").mkdir(parents=True)
    (tmp_path / "labels" / "all").mkdir(parents=True)

    # Create dummy images (small valid PNGs)
    import struct
    import zlib

    def make_png(width=64, height=64):
        """Generate a minimal valid PNG file in memory."""

        def create_chunk(chunk_type, data):
            chunk = chunk_type + data
            return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)

        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        raw_data = b''
        for _ in range(height):
            raw_data += b'\x00' + b'\xff\x00\x00' * width  # Red pixels

        idat_data = zlib.compress(raw_data)

        png = b'\x89PNG\r\n\x1a\n'
        png += create_chunk(b'IHDR', ihdr_data)
        png += create_chunk(b'IDAT', idat_data)
        png += create_chunk(b'IEND', b'')
        return png

    png_bytes = make_png()

    for i in range(10):
        # Create image
        img_path = tmp_path / "images" / "all" / f"frame_{i:04d}.png"
        img_path.write_bytes(png_bytes)

        # Create label
        label_path = tmp_path / "labels" / "all" / f"frame_{i:04d}.txt"
        label_path.write_text(sample_label_content)

    return tmp_path


@pytest.fixture
def sample_xml_content():
    """Minimal CVAT XML annotation content for testing."""
    return '''<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>1</id>
      <name>test</name>
      <size>10</size>
      <mode>interpolation</mode>
      <labels>
        <label><name>car</name></label>
        <label><name>minivan</name></label>
      </labels>
    </task>
  </meta>
  <track id="0" label="car">
    <box frame="0" outside="0" occluded="0" xtl="100" ytl="200" xbr="300" ybr="400" />
    <box frame="1" outside="0" occluded="0" xtl="110" ytl="210" xbr="310" ybr="410" />
  </track>
  <track id="1" label="minivan">
    <box frame="0" outside="0" occluded="0" xtl="500" ytl="300" xbr="700" ybr="500" />
  </track>
</annotations>'''


@pytest.fixture
def temp_cvat_dataset(tmp_path, sample_xml_content):
    """Create a temporary CVAT-style dataset for testing the converter."""
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(sample_xml_content)

    import struct
    import zlib

    def make_png(width=1920, height=1080):
        def create_chunk(chunk_type, data):
            chunk = chunk_type + data
            return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)

        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        raw_data = b''
        for _ in range(height):
            raw_data += b'\x00' + b'\x00\x00\x00' * width
        idat_data = zlib.compress(raw_data)

        png = b'\x89PNG\r\n\x1a\n'
        png += create_chunk(b'IHDR', ihdr_data)
        png += create_chunk(b'IDAT', idat_data)
        png += create_chunk(b'IEND', b'')
        return png

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    png_bytes = make_png()
    for i in range(2):
        (images_dir / f"frame_{i:04d}.PNG").write_bytes(png_bytes)

    return tmp_path, xml_path, images_dir


# ── FastAPI Test Client ──

@pytest.fixture
def test_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    os.environ['DATABASE_URL'] = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'
    os.environ['UPLOAD_DIR'] = str(tmp_path / 'uploads')
    os.environ['RESULTS_DIR'] = str(tmp_path / 'results')

    from backend.app.database import Base, engine
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def api_client(test_db):
    """FastAPI test client with a clean database."""
    from fastapi.testclient import TestClient

    from backend.app.main import app
    client = TestClient(app)
    return client
