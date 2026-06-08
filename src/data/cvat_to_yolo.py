"""
CVAT XML to YOLO Format Converter
==================================
Converts CVAT 1.1 interpolation-mode XML annotations to YOLO format.

CVAT XML format:
    <track id="0" label="car">
        <box frame="27" xtl="592.83" ytl="0.00" xbr="617.98" ybr="12.08" outside="0" .../>
    </track>

YOLO format (per-image .txt file):
    class_id x_center y_center width height  (all normalized 0-1)

Usage:
    converter = CVATToYOLOConverter(
        xml_path="datasets/raw/annotations.xml",
        images_dir="datasets/raw/images",
        output_dir="datasets/processed"
    )
    stats = converter.convert()
"""

import xml.etree.ElementTree as ET
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Single bounding box annotation."""
    frame: int
    track_id: int
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    outside: bool
    occluded: bool
    keyframe: bool


@dataclass
class ConversionStats:
    """Statistics from the conversion process."""
    total_frames: int = 0
    frames_with_annotations: int = 0
    frames_without_annotations: int = 0
    total_boxes: int = 0
    skipped_outside: int = 0
    skipped_no_image: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    boxes_per_frame: Dict[int, int] = field(default_factory=dict)


class CVATToYOLOConverter:
    """
    Converts CVAT XML annotations to YOLO format.

    The CVAT XML uses track-based interpolation annotations where each track
    represents a single object across multiple frames. This converter:

    1. Parses all tracks and their bounding boxes
    2. Groups boxes by frame number
    3. Filters out 'outside' boxes (object left the frame)
    4. Converts absolute pixel coordinates to normalized YOLO format
    5. Writes per-frame .txt label files
    6. Copies corresponding images to the output directory

    Why this approach:
    - CVAT tracks contain interpolated positions between keyframes
    - Each frame may have boxes from multiple tracks
    - 'outside=1' means the object has left the frame and should be excluded
    - YOLO expects normalized coordinates relative to image dimensions
    """

    # Class name to YOLO class ID mapping
    CLASS_MAP = {
        'car': 0,
        'minivan': 1,
    }

    def __init__(
        self,
        xml_path: str,
        images_dir: str,
        output_dir: str,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        """
        Args:
            xml_path: Path to CVAT annotations.xml file.
            images_dir: Path to directory containing frame images.
            output_dir: Path to output directory for YOLO formatted data.
                        Will create images/ and labels/ subdirectories.
            image_width: Width of the source images in pixels.
            image_height: Height of the source images in pixels.
        """
        self.xml_path = Path(xml_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.image_width = image_width
        self.image_height = image_height

        # Validate inputs exist
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

    def parse_xml(self) -> Dict[int, List[BoundingBox]]:
        """
        Parse the CVAT XML file and group bounding boxes by frame number.

        Returns:
            Dictionary mapping frame numbers to lists of BoundingBox objects.
        """
        logger.info(f"Parsing XML: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Extract metadata
        meta = root.find('.//meta/task')
        if meta is not None:
            orig_size = meta.find('original_size')
            if orig_size is not None:
                self.image_width = int(orig_size.find('width').text)
                self.image_height = int(orig_size.find('height').text)
                logger.info(f"Image size from XML: {self.image_width}x{self.image_height}")

        # Parse all tracks
        frames: Dict[int, List[BoundingBox]] = {}
        total_tracks = 0
        total_boxes = 0

        for track in root.findall('track'):
            track_id = int(track.get('id'))
            label = track.get('label')
            total_tracks += 1

            if label not in self.CLASS_MAP:
                logger.warning(f"Unknown label '{label}' in track {track_id}, skipping")
                continue

            for box in track.findall('box'):
                total_boxes += 1
                bbox = BoundingBox(
                    frame=int(box.get('frame')),
                    track_id=track_id,
                    label=label,
                    xtl=float(box.get('xtl')),
                    ytl=float(box.get('ytl')),
                    xbr=float(box.get('xbr')),
                    ybr=float(box.get('ybr')),
                    outside=box.get('outside') == '1',
                    occluded=box.get('occluded') == '1',
                    keyframe=box.get('keyframe') == '1',
                )

                if bbox.frame not in frames:
                    frames[bbox.frame] = []
                frames[bbox.frame].append(bbox)

        logger.info(f"Parsed {total_tracks} tracks, {total_boxes} total boxes, "
                     f"{len(frames)} unique frames")
        return frames

    def bbox_to_yolo(self, bbox: BoundingBox) -> Tuple[int, float, float, float, float]:
        """
        Convert a CVAT bounding box to YOLO format.

        CVAT uses absolute pixel coordinates (xtl, ytl, xbr, ybr).
        YOLO uses normalized center coordinates and dimensions:
            class_id x_center y_center width height

        All values are clipped to [0, 1] to handle edge cases where
        annotations slightly exceed image boundaries.

        Args:
            bbox: BoundingBox object with absolute pixel coordinates.

        Returns:
            Tuple of (class_id, x_center, y_center, width, height).
        """
        class_id = self.CLASS_MAP[bbox.label]

        # Clamp coordinates to image boundaries
        xtl = max(0, min(bbox.xtl, self.image_width))
        ytl = max(0, min(bbox.ytl, self.image_height))
        xbr = max(0, min(bbox.xbr, self.image_width))
        ybr = max(0, min(bbox.ybr, self.image_height))

        # Convert to YOLO format (normalized center + dimensions)
        x_center = (xtl + xbr) / 2.0 / self.image_width
        y_center = (ytl + ybr) / 2.0 / self.image_height
        width = (xbr - xtl) / self.image_width
        height = (ybr - ytl) / self.image_height

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return class_id, x_center, y_center, width, height

    def _get_image_filename(self, frame_number: int) -> Optional[str]:
        """
        Find the image file for a given frame number.

        Supports naming patterns:
            - frame_000000.PNG
            - frame_000000.jpg
            - 000000.png
        """
        patterns = [
            f"frame_{frame_number:06d}.PNG",
            f"frame_{frame_number:06d}.png",
            f"frame_{frame_number:06d}.jpg",
            f"frame_{frame_number:06d}.jpeg",
            f"{frame_number:06d}.png",
            f"{frame_number:06d}.jpg",
        ]
        for pattern in patterns:
            if (self.images_dir / pattern).exists():
                return pattern
        return None

    def convert(self) -> ConversionStats:
        """
        Run the full conversion pipeline.

        1. Parse XML annotations
        2. For each frame with annotations:
            a. Skip if no corresponding image file exists
            b. Filter out 'outside' boxes
            c. Convert remaining boxes to YOLO format
            d. Write label .txt file
            e. Copy image to output directory

        Returns:
            ConversionStats with detailed conversion metrics.
        """
        stats = ConversionStats()
        frames = self.parse_xml()

        # Create output directories
        images_out = self.output_dir / 'images' / 'all'
        labels_out = self.output_dir / 'labels' / 'all'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Determine total available frames from image files
        available_images = list(self.images_dir.glob('*.PNG')) + \
                          list(self.images_dir.glob('*.png')) + \
                          list(self.images_dir.glob('*.jpg'))
        stats.total_frames = len(available_images)
        logger.info(f"Found {stats.total_frames} image files in {self.images_dir}")

        # Process each frame
        for frame_num, boxes in sorted(frames.items()):
            image_filename = self._get_image_filename(frame_num)

            if image_filename is None:
                stats.skipped_no_image += 1
                continue

            # Filter out 'outside' boxes (object has left the frame)
            visible_boxes = [b for b in boxes if not b.outside]
            outside_count = len(boxes) - len(visible_boxes)
            stats.skipped_outside += outside_count

            if not visible_boxes:
                stats.frames_without_annotations += 1
                continue

            stats.frames_with_annotations += 1
            stats.boxes_per_frame[frame_num] = len(visible_boxes)

            # Convert boxes to YOLO format and write label file
            label_filename = Path(image_filename).stem + '.txt'
            label_lines = []

            for bbox in visible_boxes:
                class_id, xc, yc, w, h = self.bbox_to_yolo(bbox)
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                # Update class distribution
                stats.class_distribution[bbox.label] = \
                    stats.class_distribution.get(bbox.label, 0) + 1
                stats.total_boxes += 1

            # Write label file
            with open(labels_out / label_filename, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')

            # Copy image file
            src_image = self.images_dir / image_filename
            dst_image = images_out / image_filename
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

        # Log summary
        logger.info("=" * 60)
        logger.info("CONVERSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total image files:        {stats.total_frames}")
        logger.info(f"Frames with annotations:  {stats.frames_with_annotations}")
        logger.info(f"Frames without (skipped): {stats.frames_without_annotations}")
        logger.info(f"Skipped (no image file):  {stats.skipped_no_image}")
        logger.info(f"Skipped outside boxes:    {stats.skipped_outside}")
        logger.info(f"Total YOLO boxes written: {stats.total_boxes}")
        logger.info(f"Class distribution:       {stats.class_distribution}")

        if stats.boxes_per_frame:
            counts = list(stats.boxes_per_frame.values())
            logger.info(f"Boxes per frame — min: {min(counts)}, "
                         f"max: {max(counts)}, "
                         f"avg: {sum(counts)/len(counts):.1f}")

        return stats


def main():
    """CLI entry point for CVAT to YOLO conversion."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert CVAT XML to YOLO format')
    parser.add_argument('--xml', type=str,
                        default='datasets/raw/annotations.xml',
                        help='Path to CVAT annotations.xml')
    parser.add_argument('--images', type=str,
                        default='datasets/raw/images',
                        help='Path to images directory')
    parser.add_argument('--output', type=str,
                        default='datasets/processed',
                        help='Output directory for YOLO formatted data')
    args = parser.parse_args()

    converter = CVATToYOLOConverter(
        xml_path=args.xml,
        images_dir=args.images,
        output_dir=args.output,
    )
    stats = converter.convert()
    print(f"\nDone! {stats.total_boxes} boxes across {stats.frames_with_annotations} frames.")


if __name__ == '__main__':
    main()
