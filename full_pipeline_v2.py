from __future__ import annotations

import argparse
import os
import re
import time

from loguru import logger
import numpy as np
from PIL import Image

from module import LayoutRecognizer, TableStructureRecognizer, init_in_out
from module.ocr import OCR


def configure_logging(log_dir: str = "log") -> None:
    """Configure loguru logging sink to write to a log file.

    Args:
        log_dir: Directory where the log file will be saved.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "full_pipeline.log")
    logger.add(
        log_file,
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8",
        level="INFO",
    )


def _get_table_bbox(table_region: dict[str, object]) -> tuple[int, int, int, int]:
    """Extract bbox coordinate bounds of the table region.

    Args:
        table_region: Bounding box dictionary of the table region.

    Returns:
        A tuple of (x0, y0, x1, y1) as integers.
    """
    bbox = table_region.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
    return (
        int(table_region.get("x0", 0)),
        int(table_region.get("top", 0)),
        int(table_region.get("x1", 0)),
        int(table_region.get("bottom", 0)),
    )


def _gather_components(
    kwd: str,
    tb_cpns: list[dict[str, object]],
    boxes: list[dict[str, object]],
    *,
    fzy: int = 10,
    ption: float = 0.6,
) -> list[dict[str, object]]:
    """Gather and clean table components matching a specific pattern.

    Args:
        kwd: Regex pattern to match component labels.
        tb_cpns: List of detected table components.
        boxes: List of sorted text boxes.
        fzy: Fuzzy sort threshold.
        ption: Coverage threshold.

    Returns:
        Sorted list of matching and cleaned table components.
    """
    eles = LayoutRecognizer.sort_Y_firstly(
        [r for r in tb_cpns if re.match(kwd, str(r.get("label", "")))], fzy
    )
    cleaned = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
    return LayoutRecognizer.sort_Y_firstly(cleaned, 0)


def _map_box_to_rows_and_headers(
    b: dict[str, object],
    rows: list[dict[str, object]],
    headers: list[dict[str, object]],
) -> None:
    """Map box to corresponding row and header indices.

    Args:
        b: OCR box dictionary.
        rows: Sorted rows.
        headers: Sorted headers.
    """
    ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
    if ii is not None:
        b["R"] = ii
        b["R_top"] = rows[ii]["top"]
        b["R_bott"] = rows[ii]["bottom"]

    ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
    if ii is not None:
        b["H_top"] = headers[ii]["top"]
        b["H_bott"] = headers[ii]["bottom"]
        b["H_left"] = headers[ii]["x0"]
        b["H_right"] = headers[ii]["x1"]
        b["H"] = ii


def _map_box_to_columns_and_spans(
    b: dict[str, object],
    clmns: list[dict[str, object]],
    spans: list[dict[str, object]],
) -> None:
    """Map box to corresponding column and span indices.

    Args:
        b: OCR box dictionary.
        clmns: Cleaned columns.
        spans: Cleaned spans.
    """
    ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
    if ii is not None:
        b["C"] = ii
        b["C_left"] = clmns[ii]["x0"]
        b["C_right"] = clmns[ii]["x1"]

    ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
    if ii is not None:
        b["H_top"] = spans[ii]["top"]
        b["H_bott"] = spans[ii]["bottom"]
        b["H_left"] = spans[ii]["x0"]
        b["H_right"] = spans[ii]["x1"]
        b["SP"] = ii


def _process_table_mappings(
    boxes: list[dict[str, object]],
    tb_cpns: list[dict[str, object]],
) -> None:
    """Map OCR text boxes to table layout coordinates.

    Args:
        boxes: Sorted OCR text boxes.
        tb_cpns: Raw components from TableStructureRecognizer.
    """
    headers = _gather_components(r".*header$", tb_cpns, boxes)
    rows = _gather_components(r".* (row|header)", tb_cpns, boxes)
    spans = _gather_components(r".*spanning", tb_cpns, boxes)
    
    clmns = sorted(
        [r for r in tb_cpns if re.match(r"table column$", str(r.get("label", "")))],
        key=lambda x: float(x.get("x0", 0.0))
    )
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        _map_box_to_rows_and_headers(b, rows, headers)
        _map_box_to_columns_and_spans(b, clmns, spans)


def extract_table_markdown(
    img: Image.Image,
    table_region: dict[str, object],
    ocr: OCR,
) -> str:
    """Extract table markdown from table region image.

    Args:
        img: Original PIL Image.
        table_region: Bounding box dictionary of the table region.
        ocr: Initialized OCR model instance.

    Returns:
        Markdown string representing the table.
    """
    x0, y0, x1, y1 = _get_table_bbox(table_region)
    table_img = img.crop((x0, y0, x1, y1))
    
    tb_cpns = TableStructureRecognizer()([table_img])[0]
    boxes_raw = ocr(np.array(table_img))
    
    mean_h = np.mean([b[-1][1] - b[0][1] for b, _ in boxes_raw]) if boxes_raw else 1.0
    boxes = LayoutRecognizer.sort_Y_firstly(
        [
            {
                "x0": b[0][0], "x1": b[1][0],
                "top": b[0][1], "text": t[0],
                "bottom": b[-1][1],
                "layout_type": "table",
                "page_number": 0
            } 
            for b, t in boxes_raw 
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ],
        mean_h / 3
    )

    _process_table_mappings(boxes, tb_cpns)
    markdown: str = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown


def _has_table_overlap(
    box: tuple[float, float, float, float],
    exclude_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Check if the OCR box overlaps significantly with any excluded table region.

    Args:
        box: OCR box bounds (bx0, by0, bx1, by1).
        exclude_bboxes: Bounding boxes of tables.

    Returns:
        True if overlap is greater than 30%, False otherwise.
    """
    bx0, by0, bx1, by1 = box
    for lx0, ly0, lx1, ly1 in exclude_bboxes:
        ix0 = max(bx0, lx0)
        iy0 = max(by0, ly0)
        ix1 = min(bx1, lx1)
        iy1 = min(by1, ly1)

        if ix0 < ix1 and iy0 < iy1:
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            box_area = (bx1 - bx0) * (by1 - by0)
            if box_area > 0 and (inter_area / box_area) > 0.3:
                return True
    return False


def _build_markdown_from_items(merged_items: list[tuple[str, str]]) -> str:
    """Group consecutive text lines and join everything into markdown.

    Args:
        merged_items: List of (type, content) tuples.

    Returns:
        Concatenated markdown.
    """
    document_parts: list[str] = []
    current_text_block: list[str] = []

    for item_type, content in merged_items:
        if item_type == "text":
            current_text_block.append(content)
        else:
            if current_text_block:
                document_parts.append("\n".join(current_text_block))
                current_text_block = []
            document_parts.append(content)

    if current_text_block:
        document_parts.append("\n".join(current_text_block))

    return "\n\n".join(document_parts)


class DocumentProcessor:
    """Document Processor to run layout recognition and OCR on images/PDFs.

    Attributes:
        threshold: Layout detection confidence threshold.
        layout_recognizer: Recognizer for layout structures.
        ocr: OCR model instance.
    """

    threshold: float
    layout_recognizer: LayoutRecognizer
    ocr: OCR

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize models.

        Args:
            threshold: Layout detection confidence threshold. Defaults to 0.5.
        """
        self.threshold = threshold
        self.layout_recognizer = LayoutRecognizer("layout")
        self.ocr = OCR()

    def process_image(self, img: Image.Image) -> str:
        """Process a single document image and return OCR markdown.

        Args:
            img: Input PIL Image.

        Returns:
            The extracted markdown text.
        """
        logger.info("Processing image...")
        start_time = time.time()

        layouts = self.layout_recognizer.forward([img], thr=self.threshold)[0]
        logger.info(f"Detected {len(layouts)} layout regions")

        # 1. Process table layouts
        detected_tables, exclude_bboxes = self._extract_tables(img, layouts)

        # 2. Process page OCR
        kept_ocr_lines = self._process_page_ocr(img, exclude_bboxes)

        # 3. Merge and format markdown
        markdown_concat = self._merge_and_format_markdown(detected_tables, kept_ocr_lines)

        elapsed = time.time() - start_time
        logger.info(f"Processing done in {elapsed:.2f} seconds")
        return markdown_concat

    def _extract_tables(
        self,
        img: Image.Image,
        layouts: list[dict[str, object]],
    ) -> tuple[list[tuple[float, str]], list[tuple[int, int, int, int]]]:
        """Extract table regions and their bounding boxes.

        Args:
            img: Input PIL Image.
            layouts: List of detected layout dictionaries.

        Returns:
            A tuple of (detected_tables, exclude_bboxes).
        """
        detected_tables: list[tuple[float, str]] = []
        exclude_bboxes: list[tuple[int, int, int, int]] = []

        for region in layouts:
            label = str(region.get("type", "")).lower()
            score = float(region.get("score", 1.0))
            if label != "table" or score < self.threshold:
                continue

            x0, y0, x1, y1 = _get_table_bbox(region)
            exclude_bboxes.append((x0, y0, x1, y1))

            logger.info(f"Extracting table markdown for region: {region}")
            markdown = extract_table_markdown(img, region, self.ocr)
            detected_tables.append((float(y0), markdown))

        # Sort tables by their y position
        detected_tables.sort(key=lambda x: x[0])
        return detected_tables, exclude_bboxes

    def _process_page_ocr(
        self,
        img: Image.Image,
        exclude_bboxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[float, str]]:
        """Run OCR on the whole image and filter out overlapping table regions.

        Args:
            img: Input PIL Image.
            exclude_bboxes: Bounding boxes of tables to exclude.

        Returns:
            Sorted list of kept OCR lines (y_center, text).
        """
        ocr_results = self.ocr(np.array(img))
        kept_ocr_lines: list[tuple[float, str]] = []

        for b, t in ocr_results:
            if not t or not t[0]:
                continue

            bx0 = min(p[0] for p in b)
            by0 = min(p[1] for p in b)
            bx1 = max(p[0] for p in b)
            by1 = max(p[1] for p in b)

            if _has_table_overlap((bx0, by0, bx1, by1), exclude_bboxes):
                continue

            y_center = (by0 + by1) / 2
            kept_ocr_lines.append((y_center, str(t[0])))

        return kept_ocr_lines

    def _merge_and_format_markdown(
        self,
        detected_tables: list[tuple[float, str]],
        kept_ocr_lines: list[tuple[float, str]],
    ) -> str:
        """Merge OCR text and table layouts into structured markdown.

        Args:
            detected_tables: Tables sorted by Y position.
            kept_ocr_lines: OCR text lines.

        Returns:
            Formatted final markdown content.
        """
        merged_items: list[tuple[str, str]] = []
        layout_idx = 0
        num_layouts = len(detected_tables)

        for y_center, text in kept_ocr_lines:
            # Insert layouts positioned above the current OCR line
            while layout_idx < num_layouts and detected_tables[layout_idx][0] <= y_center:
                merged_items.append(("layout", detected_tables[layout_idx][1]))
                layout_idx += 1
            merged_items.append(("text", text))

        # Append remaining layouts
        while layout_idx < num_layouts:
            merged_items.append(("layout", detected_tables[layout_idx][1]))
            layout_idx += 1

        return _build_markdown_from_items(merged_items)


def process_document(
    file_path: str,
    *,
    threshold: float = 0.5,
) -> list[str]:
    """Process a document file (PDF or image) and return a list of extracted markdown texts.

    Args:
        file_path: Input file path (PDF or image).
        threshold: Confidence threshold for layout detection. Defaults to 0.5.

    Returns:
        A list of markdown strings, one for each page/image.
    """
    configure_logging()

    # Pack arguments for init_in_out compatibility
    class PipelineArgs:
        def __init__(self, inputs_path: str) -> None:
            self.inputs = inputs_path
            # init_in_out requires output_dir to exist, we can use a dummy/temporary path
            self.output_dir = "./temp_table_markdown_outputs"

    args = PipelineArgs(file_path)
    images, _ = init_in_out(args)
    logger.info(f"Loaded {len(images)} pages/images from {file_path}")

    processor = DocumentProcessor(threshold=threshold)
    results: list[str] = []
    for img in images:
        markdown = processor.process_image(img)
        results.append(markdown)

    return results


# if __name__ == "__main__":
#     input_file = r"E:\download\kms difficult file\130628 254 CBDK cu nhan su du tuyen vao cac vi tri chu chottrong NSRP giai doan II.pdf"
#     if not os.path.exists(input_file):
#         input_file = "page3_original.jpg"
        
#     logger.info(f"Running pipeline on: {input_file}")
#     extracted_texts = process_document(input_file, threshold=0.5)
    
#     for idx, text in enumerate(extracted_texts):
#         logger.info(f"--- Page {idx + 1} Extracted Text (Length: {len(text)}) ---")
#         # Print a preview
#         print(text[:500] + "\n..." if len(text) > 500 else text)
