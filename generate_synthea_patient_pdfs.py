#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

ACCENT = colors.HexColor("#0F4C81")
ACCENT_SOFT = colors.HexColor("#EEF4FA")
TEXT = colors.HexColor("#1F2937")
TEXT_MUTED = colors.HexColor("#6B7280")
BORDER = colors.HexColor("#D7DEE7")
SUCCESS = colors.HexColor("#0F766E")
WARN = colors.HexColor("#B45309")
DANGER = colors.HexColor("#B91C1C")

LAB_KEYWORDS = [
    "glucose",
    "cholesterol",
    "hdl",
    "ldl",
    "triglyceride",
    "hemoglobin",
    "hematocrit",
    "platelet",
    "leukocyte",
    "white blood cell",
    "red blood cell",
    "creatinine",
    "potassium",
    "sodium",
    "calcium",
    "bilirubin",
    "albumin",
    "protein",
    "urea",
    "bun",
    "blood pressure",
    "heart rate",
    "oxygen saturation",
    "body mass index",
    "body weight",
    "body height",
    "respiratory rate",
]

REFERENCE_RULES = [
    (["glucose"], 70, 99),
    (["cholesterol"], 0, 200),
    (["hdl"], 40, 80),
    (["ldl"], 0, 100),
    (["triglyceride"], 0, 150),
    (["hemoglobin"], 12, 17.5),
    (["hematocrit"], 36, 53),
    (["platelet"], 150, 450),
    (["white blood cell", "leukocyte"], 4, 11),
    (["red blood cell"], 4.2, 6.1),
    (["creatinine"], 0.6, 1.3),
    (["potassium"], 3.5, 5.1),
    (["sodium"], 135, 145),
    (["calcium"], 8.5, 10.5),
    (["albumin"], 3.5, 5.2),
    (["bilirubin"], 0.1, 1.2),
    (["oxygen saturation"], 95, 100),
    (["heart rate"], 60, 100),
    (["respiratory rate"], 12, 20),
    (["body mass index"], 18.5, 24.9),
    (["body weight"], 50, 100),
    (["body height"], 150, 195),
    (["systolic blood pressure"], 90, 120),
    (["diastolic blood pressure"], 60, 80),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate professional-looking synthetic patient PDFs from Synthea CSV exports."
    )
    parser.add_argument("--csv-dir", required=True, help="Path to Synthea csv directory")
    parser.add_argument("--out-dir", default="synthea_patient_reports", help="Output directory")
    parser.add_argument("--patients", type=int, default=10, help="Number of patients to export")
    parser.add_argument("--mode", choices=["clean", "scanned", "both"], default="both")
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Generate only one patient report using this PATIENT/Id value",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    if required:
        raise KeyError(f"Could not find any of columns {list(aliases)} in {list(df.columns)}")
    return None


def to_dt(value: str) -> pd.Timestamp:
    if value in (None, "", "nan"):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=False)


def format_date(value: pd.Timestamp | str | None) -> str:
    if value is None or value is pd.NaT:
        return "-"
    if isinstance(value, str):
        ts = to_dt(value)
    else:
        ts = value
    if pd.isna(ts):
        return "-"
    return ts.strftime("%d %b %Y")


def calc_age(birthdate: str, ref_dt: pd.Timestamp | None = None) -> str:
    birth = to_dt(birthdate)
    if pd.isna(birth):
        return "-"
    ref_dt = ref_dt if ref_dt is not None and not pd.isna(ref_dt) else pd.Timestamp.today()
    years = int((ref_dt.date() - birth.date()).days / 365.25)
    return str(max(years, 0))


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return value.strip("_") or "patient"


def safe_float(value: str) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def format_result(value: str) -> str:
    num = safe_float(value)
    if num is None:
        return str(value).strip() or "-"
    if abs(num) >= 100 or float(num).is_integer():
        return str(int(round(num)))
    if abs(num) >= 10:
        return f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{num:.2f}".rstrip("0").rstrip(".")


def get_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=ACCENT,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            textColor=TEXT,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11,
            textColor=TEXT_MUTED,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=ACCENT,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Cell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11,
            textColor=TEXT,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "CellCenter",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11,
            alignment=TA_CENTER,
            textColor=TEXT,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "CellHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.white,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "Label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.1,
            leading=10,
            textColor=TEXT_MUTED,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
    )
    return styles


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text or "-")), style)


def maybe_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def seed_from_text(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def create_logo(path: Path, facility_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (1200, 320), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, 1199, 319), radius=32, fill=(238, 244, 250, 255))
    draw.rounded_rectangle((36, 36, 258, 258), radius=44, fill=(15, 76, 129, 255))
    draw.rounded_rectangle((118, 68, 176, 226), radius=10, fill=(255, 255, 255, 255))
    draw.rounded_rectangle((74, 112, 220, 170), radius=10, fill=(255, 255, 255, 255))
    title_font = maybe_font(64, bold=True)
    sub_font = maybe_font(30, bold=False)
    draw.text((320, 70), facility_name[:28], font=title_font, fill=(15, 76, 129, 255))
    draw.text((322, 158), "Laboratory and Clinical Diagnostics", font=sub_font, fill=(80, 97, 117, 255))
    img.save(path)


def create_signature(path: Path, signer_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (1200, 320), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed_from_text(signer_name))
    points = []
    base_y = 130
    x = 50
    for _ in range(22):
        x += rng.randint(30, 52)
        y = base_y + rng.randint(-55, 48)
        points.append((x, y))
    draw.line(points, fill=(28, 56, 91, 255), width=6, joint="curve")
    draw.line([(65, 240), (1120, 240)], fill=(170, 184, 204, 255), width=2)
    draw.text((70, 250), signer_name, font=maybe_font(28, bold=True), fill=(31, 41, 55, 255))
    draw.text((70, 282), "Validated electronically", font=maybe_font(22), fill=(107, 114, 128, 255))
    img.save(path)


def create_stamp(path: Path, label: str = "LAB VERIFIED") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (700, 700), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (185, 28, 28, 185)
    draw.ellipse((40, 40, 660, 660), outline=color, width=12)
    draw.ellipse((90, 90, 610, 610), outline=color, width=4)
    draw.ellipse((310, 130, 390, 210), fill=color)
    font_big = maybe_font(52, bold=True)
    font_small = maybe_font(32, bold=True)
    bbox = draw.textbbox((0, 0), label, font=font_big)
    draw.text(((700 - (bbox[2] - bbox[0])) / 2, 290), label, font=font_big, fill=color)
    date_text = datetime.now().strftime("%d %b %Y").upper()
    bbox2 = draw.textbbox((0, 0), date_text, font=font_small)
    draw.text(((700 - (bbox2[2] - bbox2[0])) / 2, 372), date_text, font=font_small, fill=color)
    draw.text((162, 450), "SYNTHETIC TEST RECORD", font=font_small, fill=color)
    img.save(path)


def create_trend_chart(observations: pd.DataFrame, out_path: Path) -> tuple[Optional[Path], str]:
    if observations.empty:
        return None, ""
    desc_col = find_col(observations, ["DESCRIPTION"])
    val_col = find_col(observations, ["VALUE"])
    units_col = find_col(observations, ["UNITS"], required=False)
    date_col = find_col(observations, ["DATE", "START"], required=False)
    df = observations.copy()
    df["__num"] = pd.to_numeric(df[val_col], errors="coerce")
    if date_col:
        df["__dt"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__dt"] = pd.NaT
    df = df.dropna(subset=["__num"])
    if df.empty:
        return None, ""

    best_desc = None
    best_count = 0
    for desc, part in df.groupby(desc_col):
        if len(part) >= 3 and len(part) > best_count:
            best_desc = desc
            best_count = len(part)
    if best_desc is None:
        return None, ""

    series = df[df[desc_col] == best_desc].sort_values("__dt").tail(8)
    values = series["__num"].tolist()
    if len(values) < 3:
        return None, ""
    units = ""
    if units_col and units_col in series:
        non_empty = [u for u in series[units_col].astype(str).tolist() if u.strip()]
        units = non_empty[0] if non_empty else ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1200, 500
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((10, 10, w - 10, h - 10), radius=26, fill=(255, 255, 255), outline=(214, 222, 231), width=2)

    title = f"Trend: {best_desc}"
    if units:
        title += f" ({units})"
    draw.text((50, 28), title, font=maybe_font(32, bold=True), fill=(15, 76, 129))

    x0, y0, x1, y1 = 70, 100, 1120, 420
    for i in range(6):
        y = y0 + ((y1 - y0) / 5) * i
        draw.line((x0, y, x1, y), fill=(232, 238, 245), width=2)
    draw.line((x0, y0, x0, y1), fill=(160, 174, 192), width=2)
    draw.line((x0, y1, x1, y1), fill=(160, 174, 192), width=2)

    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmin -= 1
        vmax += 1
    pad = (vmax - vmin) * 0.15
    vmin -= pad
    vmax += pad

    points = []
    for i, value in enumerate(values):
        x = x0 + (x1 - x0) * (i / max(len(values) - 1, 1))
        y = y1 - ((value - vmin) / (vmax - vmin)) * (y1 - y0)
        points.append((x, y))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(15, 118, 110), outline=(255, 255, 255), width=2)
        label = format_result(value)
        draw.text((x - 10, y - 34), label, font=maybe_font(18, bold=True), fill=(17, 24, 39))
        dt = series.iloc[i]["__dt"]
        x_label = dt.strftime("%d %b") if not pd.isna(dt) else f"P{i + 1}"
        draw.text((x - 24, y1 + 12), x_label, font=maybe_font(18), fill=(107, 114, 128))

    draw.line(points, fill=(15, 118, 110), width=5)
    draw.text((78, 432), f"Min {format_result(min(values))}", font=maybe_font(18), fill=(107, 114, 128))
    draw.text((230, 432), f"Max {format_result(max(values))}", font=maybe_font(18), fill=(107, 114, 128))
    img.save(out_path)
    return out_path, title


def infer_reference(description: str, units: str, raw_value: str) -> tuple[str, str]:
    desc = description.lower().strip()
    num = safe_float(raw_value)
    for keys, low, high in REFERENCE_RULES:
        if any(k in desc for k in keys):
            if num is None:
                return f"{low} - {high}", ""
            if num < low:
                return f"{low} - {high}", "L"
            if num > high:
                return f"{low} - {high}", "H"
            return f"{low} - {high}", "N"
    if num is None:
        return "See note", ""
    return "Context dependent", ""


def build_keyval_table(pairs: list[tuple[str, str]], total_width: float, styles: StyleSheet1) -> Table:
    rows = []
    for label, value in pairs:
        rows.append([
            Paragraph(f"<b>{escape(label)}</b>", styles["Label"]),
            Paragraph(escape(value or "-"), styles["BodySmall"]),
        ])
    tbl = Table(rows, colWidths=[29 * mm, total_width - 29 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, 14 * mm, A4[0] - doc.rightMargin, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7A8699"))
    canvas.drawString(doc.leftMargin, 9 * mm, "Synthetic test record generated from Synthea CSV - not for clinical use")
    canvas.drawRightString(A4[0] - doc.rightMargin, 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_patient_context(data: dict[str, pd.DataFrame], patient_row: pd.Series) -> dict:
    patients = data["patients"]
    observations = data["observations"]
    encounters = data["encounters"]
    conditions = data.get("conditions")
    organizations = data.get("organizations")
    providers = data.get("providers")

    patient_id_col = find_col(patients, ["Id", "ID"])
    patient_id = str(patient_row[patient_id_col])

    obs_patient_col = find_col(observations, ["PATIENT"])
    patient_obs = observations[observations[obs_patient_col] == patient_id].copy()

    enc_patient_col = find_col(encounters, ["PATIENT"])
    enc_start_col = find_col(encounters, ["START", "DATE"], required=False)
    patient_enc = encounters[encounters[enc_patient_col] == patient_id].copy()
    if enc_start_col:
        patient_enc["__start"] = pd.to_datetime(patient_enc[enc_start_col], errors="coerce")
        patient_enc = patient_enc.sort_values("__start", ascending=False)
    latest_enc = patient_enc.iloc[0] if not patient_enc.empty else None

    condition_names: list[str] = []
    if conditions is not None and not conditions.empty:
        cond_patient_col = find_col(conditions, ["PATIENT"])
        cond_desc_col = find_col(conditions, ["DESCRIPTION"], required=False)
        cond_start_col = find_col(conditions, ["START", "DATE"], required=False)
        patient_cond = conditions[conditions[cond_patient_col] == patient_id].copy()
        if cond_start_col:
            patient_cond["__start"] = pd.to_datetime(patient_cond[cond_start_col], errors="coerce")
            patient_cond = patient_cond.sort_values("__start", ascending=False)
        if cond_desc_col:
            condition_names = [
                c for c in patient_cond[cond_desc_col].drop_duplicates().astype(str).tolist() if c.strip()
            ][:5]

    facility_name = "Synthea Medical Center"
    facility_address = "-"
    facility_phone = "-"
    if latest_enc is not None and organizations is not None and not organizations.empty:
        org_fk = find_col(encounters, ["ORGANIZATION"], required=False)
        org_id_col = find_col(organizations, ["Id", "ID"])
        org_name_col = find_col(organizations, ["NAME"], required=False)
        org_address_col = find_col(organizations, ["ADDRESS"], required=False)
        org_city_col = find_col(organizations, ["CITY"], required=False)
        org_state_col = find_col(organizations, ["STATE"], required=False)
        org_phone_col = find_col(organizations, ["PHONE"], required=False)
        if org_fk and str(latest_enc.get(org_fk, "")).strip():
            org_match = organizations[organizations[org_id_col] == str(latest_enc[org_fk])]
            if not org_match.empty:
                org_row = org_match.iloc[0]
                facility_name = str(org_row.get(org_name_col, facility_name)) or facility_name
                address_bits = [
                    str(org_row.get(org_address_col, "")).strip() if org_address_col else "",
                    str(org_row.get(org_city_col, "")).strip() if org_city_col else "",
                    str(org_row.get(org_state_col, "")).strip() if org_state_col else "",
                ]
                facility_address = ", ".join([x for x in address_bits if x]) or "-"
                facility_phone = str(org_row.get(org_phone_col, "")).strip() if org_phone_col else "-"
                if not facility_phone:
                    facility_phone = "-"

    provider_name = "Dr. Synthea Reviewer"
    provider_specialty = "Clinical Pathology"
    if latest_enc is not None and providers is not None and not providers.empty:
        provider_fk = find_col(encounters, ["PROVIDER"], required=False)
        prov_id_col = find_col(providers, ["Id", "ID"])
        prov_name_col = find_col(providers, ["NAME"], required=False)
        prov_spec_col = find_col(providers, ["SPECIALITY", "SPECIALTY"], required=False)
        if provider_fk and str(latest_enc.get(provider_fk, "")).strip():
            prov_match = providers[providers[prov_id_col] == str(latest_enc[provider_fk])]
            if not prov_match.empty:
                prov_row = prov_match.iloc[0]
                provider_name = str(prov_row.get(prov_name_col, provider_name)) or provider_name
                provider_specialty = str(prov_row.get(prov_spec_col, provider_specialty)) or provider_specialty

    return {
        "patient_id": patient_id,
        "patient_obs": patient_obs,
        "latest_enc": latest_enc,
        "condition_names": condition_names,
        "facility_name": facility_name,
        "facility_address": facility_address,
        "facility_phone": facility_phone,
        "provider_name": provider_name,
        "provider_specialty": provider_specialty,
    }


def select_report_rows(patient_obs: pd.DataFrame) -> list[dict]:
    if patient_obs.empty:
        return []
    desc_col = find_col(patient_obs, ["DESCRIPTION"])
    value_col = find_col(patient_obs, ["VALUE"])
    units_col = find_col(patient_obs, ["UNITS"], required=False)
    date_col = find_col(patient_obs, ["DATE", "START"], required=False)
    type_col = find_col(patient_obs, ["TYPE"], required=False)

    df = patient_obs.copy()
    df["__desc_lower"] = df[desc_col].astype(str).str.lower()
    df["__date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["__value_num"] = pd.to_numeric(df[value_col], errors="coerce")

    preferred_mask = df["__desc_lower"].str.contains("|".join(re.escape(k) for k in LAB_KEYWORDS), na=False)
    preferred = df[preferred_mask].copy()
    pool = preferred if len(preferred) >= 6 else df.copy()

    if "__date" in pool:
        pool = pool.sort_values("__date", ascending=False)
    pool = pool.drop_duplicates(subset=[desc_col], keep="first")

    rows = []
    for _, row in pool.head(12).iterrows():
        desc = str(row.get(desc_col, "")).strip()
        raw_value = str(row.get(value_col, "")).strip()
        units = str(row.get(units_col, "")).strip() if units_col else ""
        kind = str(row.get(type_col, "")).strip() if type_col else ""
        ref_range, flag = infer_reference(desc, units, raw_value)
        if flag == "N":
            flag = ""
        rows.append(
            {
                "description": desc or "Observation",
                "result": format_result(raw_value),
                "units": units or "-",
                "reference": ref_range,
                "flag": flag,
                "date": format_date(row.get("__date")),
                "kind": kind or "Observation",
            }
        )
    return rows


def build_impression(report_rows: list[dict], condition_names: list[str]) -> str:
    flagged = [r for r in report_rows if r.get("flag") in {"H", "L"}]
    if flagged:
        snippets = []
        for row in flagged[:4]:
            label = "elevated" if row["flag"] == "H" else "reduced"
            snippets.append(f"{row['description']} is {label}")
        sentence = "; ".join(snippets)
        base = f"Review of the most recent measurements indicates that {sentence}. Correlation with the clinical context and longitudinal follow-up is recommended."
    else:
        base = "The selected measurements are broadly aligned with the expected reference intervals for this synthetic patient profile, without a dominant abnormal trend in the latest snapshot."
    if condition_names:
        base += " Documented background problems include " + ", ".join(condition_names[:3]) + "."
    base += " This document is synthetic and intended only for pipeline development, OCR, parsing, and retrieval evaluation."
    return base


def build_results_table(rows: list[dict], styles: StyleSheet1) -> Table:
    header = [
        Paragraph("Test / Observation", styles["CellHeader"]),
        Paragraph("Result", styles["CellHeader"]),
        Paragraph("Units", styles["CellHeader"]),
        Paragraph("Reference", styles["CellHeader"]),
        Paragraph("Flag", styles["CellHeader"]),
        Paragraph("Date", styles["CellHeader"]),
    ]
    data = [header]
    for row in rows:
        flag = row["flag"]
        flag_text = flag if flag else "-"
        if flag == "H":
            flag_html = '<font color="#B91C1C"><b>H</b></font>'
        elif flag == "L":
            flag_html = '<font color="#B45309"><b>L</b></font>'
        else:
            flag_html = escape(flag_text)
        data.append(
            [
                Paragraph(escape(row["description"]), styles["Cell"]),
                Paragraph(escape(row["result"]), styles["CellCenter"]),
                Paragraph(escape(row["units"]), styles["CellCenter"]),
                Paragraph(escape(row["reference"]), styles["CellCenter"]),
                Paragraph(flag_html, styles["CellCenter"]),
                Paragraph(escape(row["date"]), styles["CellCenter"]),
            ]
        )
    tbl = Table(
        data,
        colWidths=[58 * mm, 22 * mm, 20 * mm, 34 * mm, 12 * mm, 24 * mm],
        repeatRows=1,
    )
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    for row_idx in range(1, len(data)):
        bg = colors.white if row_idx % 2 else ACCENT_SOFT
        style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
        flag = rows[row_idx - 1]["flag"]
        if flag == "H":
            style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), DANGER))
        elif flag == "L":
            style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), WARN))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def generate_clean_pdf(data: dict[str, pd.DataFrame], patient_row: pd.Series, out_pdf: Path) -> Path:
    styles = get_styles()
    context = build_patient_context(data, patient_row)
    report_rows = select_report_rows(context["patient_obs"])
    if not report_rows:
        raise ValueError(f"No observations found for patient {context['patient_id']}")

    first_col = find_col(data["patients"], ["FIRST"], required=False)
    last_col = find_col(data["patients"], ["LAST"], required=False)
    birth_col = find_col(data["patients"], ["BIRTHDATE"], required=False)
    gender_col = find_col(data["patients"], ["GENDER"], required=False)
    race_col = find_col(data["patients"], ["RACE"], required=False)
    city_col = find_col(data["patients"], ["CITY"], required=False)
    state_col = find_col(data["patients"], ["STATE"], required=False)
    address_col = find_col(data["patients"], ["ADDRESS"], required=False)
    zip_col = find_col(data["patients"], ["ZIP"], required=False)

    latest_enc = context["latest_enc"]
    enc_class = "-"
    visit_date = pd.Timestamp.today()
    if latest_enc is not None:
        enc_class = str(latest_enc.get(find_col(data["encounters"], ["ENCOUNTERCLASS"], required=False), "")).title() or "-"
        start_col = find_col(data["encounters"], ["START", "DATE"], required=False)
        if start_col:
            visit_date = to_dt(str(latest_enc.get(start_col, "")))
            if pd.isna(visit_date):
                visit_date = pd.Timestamp.today()

    patient_name = " ".join(
        [
            str(patient_row.get(first_col, "")).strip() if first_col else "",
            str(patient_row.get(last_col, "")).strip() if last_col else "",
        ]
    ).strip()
    patient_name = patient_name or f"Patient {context['patient_id'][:8]}"
    age = calc_age(str(patient_row.get(birth_col, "")) if birth_col else "", visit_date)
    sex = str(patient_row.get(gender_col, "")).strip() if gender_col else "-"
    race = str(patient_row.get(race_col, "")).strip() if race_col else "-"
    address_parts = [
        str(patient_row.get(address_col, "")).strip() if address_col else "",
        str(patient_row.get(city_col, "")).strip() if city_col else "",
        str(patient_row.get(state_col, "")).strip() if state_col else "",
        str(patient_row.get(zip_col, "")).strip() if zip_col else "",
    ]
    address = ", ".join([a for a in address_parts if a]) or "-"

    safe_stem = slugify(f"{patient_name}_{context['patient_id'][:8]}")
    assets_dir = out_pdf.parent / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logo_path = assets_dir / f"{safe_stem}_logo.png"
    signature_path = assets_dir / f"{safe_stem}_signature.png"
    stamp_path = assets_dir / f"{safe_stem}_stamp.png"
    chart_path = assets_dir / f"{safe_stem}_trend.png"

    create_logo(logo_path, context["facility_name"])
    create_signature(signature_path, context["provider_name"])
    create_stamp(stamp_path)
    chart_image_path, chart_title = create_trend_chart(context["patient_obs"], chart_path)

    report_id = f"SR-{visit_date.strftime('%Y%m%d')}-{context['patient_id'][:8].upper()}"
    impression = build_impression(report_rows, context["condition_names"])

    patient_left = build_keyval_table(
        [
            ("Patient", patient_name),
            ("Patient ID", context["patient_id"]),
            ("Birth date", format_date(str(patient_row.get(birth_col, "")) if birth_col else "")),
            ("Age", age),
            ("Sex", sex),
            ("Race", race),
        ],
        85 * mm,
        styles,
    )
    patient_right = build_keyval_table(
        [
            ("Report ID", report_id),
            ("Visit date", format_date(visit_date)),
            ("Encounter", enc_class),
            ("Ordering clinician", context["provider_name"]),
            ("Specialty", context["provider_specialty"]),
            ("Address", address),
        ],
        85 * mm,
        styles,
    )

    header_block = Table(
        [
            [
                RLImage(str(logo_path), width=48 * mm, height=12.8 * mm),
                [
                    Paragraph(escape(context["facility_name"]), styles["ReportTitle"]),
                    Paragraph(escape(context["facility_address"]), styles["BodySmall"]),
                    Paragraph(escape(f"Phone: {context['facility_phone']}   |   Report status: Final"), styles["BodyMuted"]),
                ],
            ]
        ],
        colWidths=[52 * mm, 122 * mm],
    )
    header_block.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    info_wrapper = Table([[patient_left, patient_right]], colWidths=[87 * mm, 87 * mm])
    info_wrapper.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))

    result_table = build_results_table(report_rows, styles)

    story = [header_block, Spacer(1, 8), Paragraph("PATIENT LABORATORY AND OBSERVATION REPORT", styles["SectionTitle"]), Spacer(1, 2), info_wrapper, Spacer(1, 10)]

    story.extend(
        [
            Paragraph("Selected Results", styles["SectionTitle"]),
            result_table,
            Spacer(1, 8),
        ]
    )

    if chart_image_path:
        summary_box = Table(
            [
                [
                    [
                        Paragraph("Clinical Summary", styles["SectionTitle"]),
                        Paragraph(escape(impression), styles["BodySmall"]),
                        Spacer(1, 4),
                        Paragraph(
                            escape("Background conditions: " + (", ".join(context["condition_names"]) if context["condition_names"] else "None documented in current export.")),
                            styles["BodyMuted"],
                        ),
                    ],
                    [
                        Paragraph(chart_title or "Trend", styles["SectionTitle"]),
                        RLImage(str(chart_image_path), width=72 * mm, height=30 * mm),
                        Paragraph("Raster chart included intentionally for multimodal document testing.", styles["BodyMuted"]),
                    ],
                ]
            ],
            colWidths=[92 * mm, 82 * mm],
        )
    else:
        summary_box = Table(
            [
                [
                    Paragraph("Clinical Summary", styles["SectionTitle"]),
                    Paragraph(escape(impression), styles["BodySmall"]),
                ]
            ],
            colWidths=[174 * mm],
        )
    summary_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([summary_box, Spacer(1, 10)])

    auth_table = Table(
        [
            [
                [
                    Paragraph("Validated By", styles["SectionTitle"]),
                    Paragraph(escape(context["provider_name"]), styles["BodySmall"]),
                    Paragraph(escape(context["provider_specialty"]), styles["BodyMuted"]),
                    Spacer(1, 3),
                    RLImage(str(signature_path), width=64 * mm, height=17 * mm),
                ],
                [
                    Paragraph("Laboratory Seal", styles["SectionTitle"]),
                    RLImage(str(stamp_path), width=28 * mm, height=28 * mm),
                ],
            ]
        ],
        colWidths=[125 * mm, 49 * mm],
    )
    auth_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(KeepTogether([Paragraph("Authorization", styles["SectionTitle"]), auth_table]))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=f"Synthetic Report - {patient_name}",
        author="OpenAI / Synthea synthetic generator",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_pdf


def degrade_image_for_scan(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")

    # Slight contrast and color flattening
    img = ImageEnhance.Contrast(img).enhance(0.95)
    img = ImageEnhance.Sharpness(img).enhance(0.9)

    # Add paper background
    paper = Image.new("RGB", (int(img.width * 1.04), int(img.height * 1.04)), (246, 244, 238))
    noise = Image.effect_noise(paper.size, 6).convert("L")
    paper = Image.blend(paper, ImageOps.colorize(noise, (240, 238, 232), (255, 255, 255)), 0.18)
    paper.paste(img, (int(img.width * 0.02), int(img.height * 0.02)))
    img = paper

    # Slight skew/rotation
    angle = rng.uniform(-1.1, 1.1)
    img = img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(246, 244, 238))

    # JPEG re-compression artifact simulation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=55)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Blur + grayscale-like scan feel
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.8)))
    gray = ImageOps.grayscale(img)
    img = ImageOps.colorize(gray, black=(48, 48, 48), white=(252, 250, 245)).convert("RGB")

    # Speckle noise overlay
    noise = Image.effect_noise(img.size, rng.uniform(6, 10)).convert("L")
    noise = ImageOps.colorize(noise, (230, 228, 222), (255, 255, 255)).convert("RGB")
    img = Image.blend(img, noise, 0.10)

    return img


def make_scanned_pdf(clean_pdf: Path, scanned_pdf: Path) -> Path:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for scanned mode. Install it with: pip install pymupdf")
    doc = fitz.open(clean_pdf)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=180, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = degrade_image_for_scan(img, seed=1337 + i)
        pages.append(img)
    if not pages:
        raise ValueError(f"No pages rendered from {clean_pdf}")
    scanned_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(scanned_pdf, save_all=True, append_images=pages[1:], resolution=150.0)
    return scanned_pdf


def load_data(csv_dir: Path) -> dict[str, pd.DataFrame]:
    data = {
        "patients": read_csv(csv_dir / "patients.csv"),
        "observations": read_csv(csv_dir / "observations.csv"),
        "encounters": read_csv(csv_dir / "encounters.csv"),
    }
    optional_files = ["conditions", "organizations", "providers"]
    for name in optional_files:
        path = csv_dir / f"{name}.csv"
        if path.exists():
            data[name] = read_csv(path)
        else:
            data[name] = pd.DataFrame()
    return data


def choose_patients(data: dict[str, pd.DataFrame], limit: int, patient_id: Optional[str]) -> pd.DataFrame:
    patients = data["patients"].copy()
    id_col = find_col(patients, ["Id", "ID"])
    if patient_id:
        match = patients[patients[id_col] == patient_id]
        if match.empty:
            raise ValueError(f"Patient ID not found: {patient_id}")
        return match.head(1)

    obs = data["observations"]
    obs_patient_col = find_col(obs, ["PATIENT"])
    counts = obs.groupby(obs_patient_col).size().sort_values(ascending=False)
    chosen_ids = counts.head(limit).index.tolist()
    chosen = patients[patients[id_col].isin(chosen_ids)].copy()
    return chosen.head(limit)


def main() -> int:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    data = load_data(csv_dir)
    patient_rows = choose_patients(data, args.patients, args.patient_id)
    if patient_rows.empty:
        print("No patients selected.", file=sys.stderr)
        return 1

    generated = []
    for _, patient_row in patient_rows.iterrows():
        patient_id = str(patient_row[find_col(data["patients"], ["Id", "ID"])])
        first_col = find_col(data["patients"], ["FIRST"], required=False)
        last_col = find_col(data["patients"], ["LAST"], required=False)
        name = " ".join(
            [
                str(patient_row.get(first_col, "")).strip() if first_col else "",
                str(patient_row.get(last_col, "")).strip() if last_col else "",
            ]
        ).strip()
        stem = slugify(f"{name}_{patient_id[:8]}")
        clean_pdf = out_dir / "clean" / f"{stem}_clean.pdf"
        scanned_pdf = out_dir / "scanned" / f"{stem}_scanned.pdf"

        try:
            if args.mode in {"clean", "both", "scanned"}:
                generate_clean_pdf(data, patient_row, clean_pdf)
                generated.append(clean_pdf)
                print(f"[OK] {clean_pdf}")
            if args.mode in {"scanned", "both"}:
                make_scanned_pdf(clean_pdf, scanned_pdf)
                generated.append(scanned_pdf)
                print(f"[OK] {scanned_pdf}")
        except Exception as exc:
            print(f"[WARN] Failed for patient {patient_id}: {exc}", file=sys.stderr)

    print(f"\nGenerated {len(generated)} file(s) in {out_dir}")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
