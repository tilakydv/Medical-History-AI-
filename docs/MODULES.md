# Module reference

## OCR

`OCRService` extracts text from each PDF page with PyMuPDF. Pages with insufficient embedded
text are rendered at 2x resolution, preprocessed (grayscale, autocontrast, contrast enhancement,
median filtering), and passed to lazy-loaded PaddleOCR. The response retains method provenance
per page. Raster images always use OCR.

## Laboratory processing

`LaboratoryService` parses only lines matching a conservative test/value/unit/range shape.
Numeric values and qualitative results are stored separately. It derives H/L/N only when a
reference range is present and preserves explicit flags. Historical queries group observations
into graph-ready series without interpreting clinical significance.

## MRI

`MRIService.inspect` verifies supported inputs and returns non-PHI technical metadata.
`analyze` normalizes finite NIfTI voxel intensities, invokes the configured nnU-Net adapter,
validates mask geometry, computes physical mask volume from voxel spacing, finds bounding-box
and centroid coordinates, and creates an overlay for review. No heuristic substitutes for an
unavailable model.

## Integration

`LLMIntegrationService` emits schema-versioned dictionaries containing normalized source,
document, lab, model, segmentation, and findings fields. It performs no generation or medical
reasoning. This boundary allows Qwen-based modules to consume stable data without raw OCR.

## Error and validation behavior

Expected domain failures return `{error, message}` with suitable HTTP status codes. Uploads are
streamed in chunks, bounded by configuration, hashed during storage, checked against known file
signatures, and deduplicated per patient and type. Processing errors do not return fabricated
partial results.
