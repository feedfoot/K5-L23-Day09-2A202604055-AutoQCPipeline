"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODE = "BOX_OUT_OF_IMAGE"
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}          # cộng mọi mã bắt đầu bằng DUP_IMAGE_


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _fx(image, line, action, before, after):
    return {"image": image, "line": line, "action": action, "before": before, "after": after}


def _clip(text):
    """Cắt box về trong [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích / không đọc được."""
    try:
        c, cx, cy, w, h = parse_yolo_line(text)
    except ValueError:
        return None
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {nw:.6f} {nh:.6f}"


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]             # WARN là việc của người: không đụng
    fixes = []

    # 1) Xoá ảnh (trùng / rò rỉ test / hỏng / quá nhỏ): làm TRƯỚC để bỏ qua issue dòng của ảnh đã xoá.
    removed = set()
    for i in errors:
        if not _is_remove(i.code) or i.image in removed:
            continue
        removed.add(i.image)
        img, lab = ds.image_path(i.image), ds.label_path(i.image)
        gone = []
        if img is not None and img.exists():
            img.unlink()
            gone.append(img.name)
        if lab.exists():
            lab.unlink()
            gone.append(lab.name)
        if gone:
            fixes.append(_fx(i.image, None, "remove_image", " + ".join(gone), f"đã xoá ({i.code})"))

    # 2) Tên lớp trong data.yaml: ghi lại đúng cfg["classes"] (một lần dù có nhiều issue).
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        try:
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) if ds.yaml_path.exists() else None
        except yaml.YAMLError:
            d = None
        d = d if isinstance(d, dict) else {}
        old = d.get("names")
        d["names"] = {k: n for k, n in enumerate(cfg["classes"])}
        ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append(_fx("data.yaml", None, "rename_classes", str(old), str(d["names"])))

    # 3) Thiếu file nhãn: KHÔNG tạo file (rỗng = "ảnh nền"), chỉ ghi lại để chuyển người gán.
    needs = set()
    for i in errors:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and i.image not in needs:
            needs.add(i.image)
            fixes.append(_fx(i.image, None, "needs_label", "(không có file)", "chuyển người gán"))

    # 4) Lỗi theo dòng: gom theo file -> {số dòng GỐC: {mã lỗi}}, rồi ghi lại file MỘT lần.
    by_file = defaultdict(lambda: defaultdict(set))
    for i in errors:
        if i.line is not None and (i.code in DROP_CODES or i.code == CLIP_CODE) and i.image not in removed:
            by_file[i.image][i.line].add(i.code)

    for stem in sorted(by_file):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()   # giữ dòng trống -> số dòng không lệch
        keep, changed = [], False
        for lno, text in enumerate(raw, 1):
            codes = by_file[stem].get(lno)
            if not codes:
                keep.append(text)
                continue
            if codes & DROP_CODES:                                   # xoá thắng cắt: chỉ 1 fix drop_line
                fixes.append(_fx(stem, lno, "drop_line", text, f"(xoá: {', '.join(sorted(codes & DROP_CODES))})"))
                changed = True
                continue
            new = _clip(text)                                        # chỉ còn BOX_OUT_OF_IMAGE
            if new is None:                                          # nằm hẳn ngoài ảnh: xoá thay vì box rỗng
                fixes.append(_fx(stem, lno, "drop_line", text, "(xoá: box nằm ngoài ảnh)"))
            else:
                fixes.append(_fx(stem, lno, "clip_box", text, new))
                keep.append(new)
            changed = True
        if changed:
            path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
    return fixes
