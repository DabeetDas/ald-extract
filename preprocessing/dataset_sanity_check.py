"""
Comprehensive Dataset Sanity Check Script
==========================================
Audits the Data/ directory for quality issues across all papers.

Checks performed:
  1. Structural checks   — missing content.txt, missing Images dir
  2. Text quality checks — empty files, gibberish, very short/long text, 
                           non-ASCII ratio, excessive whitespace
  3. Image checks        — no images, corrupt/unreadable images, very small
                           images (likely artifacts), format distribution
  4. Duplicate detection — identical content.txt files (md5 hash)
  5. Statistical summary — distributions of text length, image count, etc.
"""

import os
import re
import sys
import hashlib
import statistics
from collections import Counter, defaultdict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))

# Thresholds (tune as needed)
GIBBERISH_RATIO_THRESHOLD = 0.3
SHORT_TEXT_THRESHOLD = 200          # chars
VERY_SHORT_TEXT_THRESHOLD = 50      # chars
LONG_TEXT_THRESHOLD = 500_000       # chars
HIGH_NON_ASCII_RATIO = 0.3         # >30% non-ascii chars
EXCESSIVE_WHITESPACE_RATIO = 0.5   # >50% whitespace
MIN_IMAGE_BYTES = 500              # images smaller than this are suspicious
GIBBERISH_RE = re.compile(r'(/C[0-9a-fA-F]{2,3}){3,}')


# ──────────────────────────────────────────────────
# Individual check functions
# ──────────────────────────────────────────────────

def check_text_quality(text: str) -> list[str]:
    """Returns a list of issues found in the text."""
    issues = []
    if not text.strip():
        issues.append("EMPTY_TEXT")
        return issues

    length = len(text)

    # Length checks
    if length < VERY_SHORT_TEXT_THRESHOLD:
        issues.append("VERY_SHORT_TEXT")
    elif length < SHORT_TEXT_THRESHOLD:
        issues.append("SHORT_TEXT")
    if length > LONG_TEXT_THRESHOLD:
        issues.append("VERY_LONG_TEXT")

    # Gibberish check (/CXX patterns from bad PDF extraction)
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        gibberish_lines = sum(1 for l in lines if l.strip().startswith('/C'))
        if gibberish_lines / len(lines) > GIBBERISH_RATIO_THRESHOLD:
            issues.append("GIBBERISH_TEXT")

    # Non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if length > 0 and (non_ascii / length) > HIGH_NON_ASCII_RATIO:
        issues.append("HIGH_NON_ASCII")

    # Excessive whitespace
    whitespace = sum(1 for c in text if c in (' ', '\t', '\n', '\r'))
    if length > 0 and (whitespace / length) > EXCESSIVE_WHITESPACE_RATIO:
        issues.append("EXCESSIVE_WHITESPACE")

    # Low word count (might be a table-only or formula-only extraction)
    words = text.split()
    if len(words) < 20 and length > VERY_SHORT_TEXT_THRESHOLD:
        issues.append("LOW_WORD_COUNT")

    return issues


def check_images(images_dir: str) -> tuple[list[str], dict]:
    """Check images in a directory for problems. Returns issues and stats."""
    issues = []
    stats = {"count": 0, "formats": [], "sizes": [], "corrupt": 0, "tiny": 0}

    if not os.path.isdir(images_dir):
        issues.append("NO_IMAGES_DIR")
        return issues, stats

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    stats["count"] = len(image_files)
    if len(image_files) == 0:
        issues.append("EMPTY_IMAGES_DIR")
        return issues, stats

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        file_size = os.path.getsize(img_path)
        ext = os.path.splitext(img_file)[1].lower()
        stats["formats"].append(ext)
        stats["sizes"].append(file_size)

        if file_size < MIN_IMAGE_BYTES:
            stats["tiny"] += 1

        if HAS_PIL:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                stats["corrupt"] += 1

    if stats["corrupt"] > 0:
        issues.append(f"CORRUPT_IMAGES({stats['corrupt']})")
    if stats["tiny"] > 0:
        issues.append(f"TINY_IMAGES({stats['tiny']})")

    return issues, stats


def file_hash(path: str) -> str:
    """Returns MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────
# Main audit
# ──────────────────────────────────────────────────

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    papers = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
    total = len(papers)

    # Aggregate trackers
    issue_tracker = defaultdict(list)   # issue_type -> [paper_ids]
    text_lengths = []
    image_counts = []
    image_format_counter = Counter()
    all_image_sizes = []
    hash_map = defaultdict(list)        # hash -> [paper_ids]  for duplicate detection

    print(f"Scanning {total} papers in {DATA_DIR}...\n")

    for paper in papers:
        paper_dir = os.path.join(DATA_DIR, paper)
        content_path = os.path.join(paper_dir, "content.txt")
        images_dir = os.path.join(paper_dir, "Images")

        # ── Text checks ──
        if not os.path.exists(content_path):
            issue_tracker["MISSING_CONTENT_TXT"].append(paper)
            text_lengths.append(0)
        else:
            size = os.path.getsize(content_path)
            if size == 0:
                issue_tracker["EMPTY_FILE"].append(paper)
                text_lengths.append(0)
            else:
                with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                text_lengths.append(len(text))
                text_issues = check_text_quality(text)
                for iss in text_issues:
                    issue_tracker[iss].append(paper)

                # Hash for duplicate detection
                h = file_hash(content_path)
                hash_map[h].append(paper)

        # ── Image checks ──
        img_issues, img_stats = check_images(images_dir)
        image_counts.append(img_stats["count"])
        image_format_counter.update(img_stats["formats"])
        all_image_sizes.extend(img_stats["sizes"])
        for iss in img_issues:
            issue_tracker[iss].append(paper)

    # ── Duplicate detection ──
    duplicates = {h: papers for h, papers in hash_map.items() if len(papers) > 1}

    # ──────────────────────────────────────────────────
    # Print report
    # ──────────────────────────────────────────────────

    W = 62

    print("=" * W)
    print("  DATASET SANITY CHECK REPORT")
    print("=" * W)

    # 1. Overview
    print(f"\n{'─' * W}")
    print("  1. OVERVIEW")
    print(f"{'─' * W}")
    print(f"  Total paper directories:        {total}")
    non_zero_lengths = [l for l in text_lengths if l > 0]
    print(f"  Papers with text:               {len(non_zero_lengths)}")
    print(f"  Papers without text:            {total - len(non_zero_lengths)}")
    print(f"  Total images across dataset:    {sum(image_counts)}")

    # 2. Text statistics
    print(f"\n{'─' * W}")
    print("  2. TEXT STATISTICS")
    print(f"{'─' * W}")
    if non_zero_lengths:
        print(f"  Min text length:                {min(non_zero_lengths):,} chars")
        print(f"  Max text length:                {max(non_zero_lengths):,} chars")
        print(f"  Mean text length:               {statistics.mean(non_zero_lengths):,.0f} chars")
        print(f"  Median text length:             {statistics.median(non_zero_lengths):,.0f} chars")
        print(f"  Std dev:                        {statistics.stdev(non_zero_lengths):,.0f} chars" if len(non_zero_lengths) > 1 else "")

    # 3. Image statistics
    print(f"\n{'─' * W}")
    print("  3. IMAGE STATISTICS")
    print(f"{'─' * W}")
    papers_with_imgs = sum(1 for c in image_counts if c > 0)
    print(f"  Papers with images:             {papers_with_imgs}")
    print(f"  Papers without images:          {total - papers_with_imgs}")
    if image_counts:
        non_zero_img = [c for c in image_counts if c > 0]
        if non_zero_img:
            print(f"  Min images per paper:           {min(non_zero_img)}")
            print(f"  Max images per paper:           {max(non_zero_img)}")
            print(f"  Mean images per paper:          {statistics.mean(non_zero_img):.1f}")
    print(f"  Image format distribution:")
    for fmt, cnt in image_format_counter.most_common():
        print(f"    {fmt:10s} {cnt:>6,}")
    if all_image_sizes:
        print(f"  Min image file size:            {min(all_image_sizes):,} bytes")
        print(f"  Max image file size:            {max(all_image_sizes):,} bytes")
        print(f"  Mean image file size:           {statistics.mean(all_image_sizes):,.0f} bytes")

    # 4. Issues found
    print(f"\n{'─' * W}")
    print("  4. ISSUES FOUND")
    print(f"{'─' * W}")
    if not issue_tracker:
        print("  No issues found! Dataset is clean.")
    else:
        for issue_type in sorted(issue_tracker.keys()):
            papers_list = issue_tracker[issue_type]
            count = len(papers_list)
            pct = count / total * 100
            preview = ", ".join(papers_list[:5])
            suffix = f", ... (+{count-5} more)" if count > 5 else ""
            print(f"  {issue_type:30s}  {count:>5}  ({pct:4.1f}%)  [{preview}{suffix}]")

    # 5. Duplicates
    print(f"\n{'─' * W}")
    print("  5. DUPLICATE CONTENT FILES")
    print(f"{'─' * W}")
    if not duplicates:
        print("  No duplicate content files found.")
    else:
        print(f"  Found {len(duplicates)} groups of duplicate content:")
        for h, dup_papers in list(duplicates.items())[:10]:
            print(f"    [{len(dup_papers)} copies] {', '.join(dup_papers[:6])}" +
                  (f" ... (+{len(dup_papers)-6})" if len(dup_papers) > 6 else ""))
        if len(duplicates) > 10:
            print(f"    ... and {len(duplicates) - 10} more duplicate groups")

    # 6. Summary verdict
    total_issues = sum(len(v) for v in issue_tracker.values()) + sum(len(v) - 1 for v in duplicates.values())
    total_unique_problem_papers = len(set(p for papers in issue_tracker.values() for p in papers))
    print(f"\n{'=' * W}")
    print(f"  VERDICT")
    print(f"{'=' * W}")
    print(f"  Papers with at least 1 issue:   {total_unique_problem_papers} / {total} ({total_unique_problem_papers/total*100:.1f}%)")
    print(f"  Papers fully clean:             {total - total_unique_problem_papers} / {total} ({(total - total_unique_problem_papers)/total*100:.1f}%)")
    if duplicates:
        dup_count = sum(len(v) - 1 for v in duplicates.values())
        print(f"  Duplicate papers (removable):   {dup_count}")
    print(f"{'=' * W}\n")

    # Save detailed report
    report_path = os.path.join(os.path.dirname(__file__), "sanity_report.txt")
    with open(report_path, "w") as f:
        for issue_type in sorted(issue_tracker.keys()):
            f.write(f"--- {issue_type} ---\n")
            for p in issue_tracker[issue_type]:
                f.write(f"  {p}\n")
            f.write("\n")
        if duplicates:
            f.write("--- DUPLICATES ---\n")
            for h, dup_papers in duplicates.items():
                f.write(f"  [hash={h[:8]}] {', '.join(dup_papers)}\n")
    print(f"Detailed issue list saved to: {report_path}")


if __name__ == "__main__":
    main()
