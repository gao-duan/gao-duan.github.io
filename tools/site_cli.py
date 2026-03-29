#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"
TEMPLATE_PATH = ROOT / "templates" / "index.html.tmpl"
INDEX_PATH = ROOT / "index.html"


@dataclass
class SiteData:
    site: dict[str, Any]
    home: dict[str, Any]
    publications: list[dict[str, Any]]
    thesis: list[dict[str, Any]]
    experience: list[dict[str, Any]]
    projects: list[dict[str, Any]]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_site_data() -> SiteData:
    return SiteData(
        site=load_json(CONTENT_DIR / "site.json"),
        home=load_json(CONTENT_DIR / "home.json"),
        publications=load_json(CONTENT_DIR / "publications.json"),
        thesis=load_json(CONTENT_DIR / "thesis.json"),
        experience=load_json(CONTENT_DIR / "experience.json"),
        projects=load_json(CONTENT_DIR / "projects.json"),
    )


def is_external(href: str) -> bool:
    return href.startswith(("http://", "https://", "mailto:"))


def local_path(href: str) -> Path:
    clean = href.split("?", 1)[0].split("#", 1)[0]
    return ROOT / clean


def attrs_to_html(attrs: dict[str, str | None]) -> str:
    parts: list[str] = []
    for key, value in attrs.items():
        if value is None:
            continue
        parts.append(f'{key}="{escape(value, quote=True)}"')
    return " ".join(parts)


def render_link(link: dict[str, Any], class_name: str = "btn myicon") -> str:
    attrs = {
        "href": str(link["href"]),
        "class": class_name,
        "target": "_blank",
        "rel": "noopener noreferrer",
    }
    return f'<a {attrs_to_html(attrs)}>{escape(link["label"])}</a>'


def render_nav(site: dict[str, Any]) -> str:
    items = []
    for item in site["navigation"]:
        items.append(
            "          <li class=\"nav-item\">"
            f"<a class=\"nav-link\" href=\"{escape(item['href'], quote=True)}\">{escape(item['label'])}</a>"
            "</li>"
        )
    return "\n".join(items)


def render_contact_links(site: dict[str, Any], home: dict[str, Any]) -> str:
    links = list(home["contact_links"])
    cv = site["cv"]
    links.insert(
        2,
        {
            "label": f"{cv['label']} {cv['year']}",
            "href": cv["href"],
            "icon_class": "ai ai-cv",
            "aria_label": f"Curriculum vitae ({cv['year']})",
            "external": False,
        },
    )

    rendered = []
    for link in links:
        attrs = {
            "href": str(link["href"]),
            "class": "contact-link",
            "aria-label": str(link["aria_label"]),
        }
        if link.get("external", False) or str(link["href"]).startswith(("http://", "https://")):
            attrs["target"] = "_blank"
            attrs["rel"] = "noopener noreferrer"
        rendered.append(
            f'            <a {attrs_to_html(attrs)}>'
            f'<i class="{escape(link["icon_class"], quote=True)}" aria-hidden="true"></i>'
            f'<span class="sr-only">{escape(link["label"])}</span>'
            "</a>"
        )
    return "\n".join(rendered)


def render_bio_paragraphs(home: dict[str, Any]) -> str:
    return "\n".join(
        f'          <p class="my-3 text-large-2">{paragraph}</p>'
        for paragraph in home["profile"]["intro_paragraphs"]
    )


def render_publications(publications: list[dict[str, Any]]) -> str:
    chunks = []
    for publication in publications:
        links_html = "\n".join(
            f"              {render_link(link)}" for link in publication["links"]
        )
        chunks.append(
            f"""      <article class="row publication-card" id="publication-{escape(publication['slug'], quote=True)}">
        <div class="col-md-3">
          <img src="{escape(publication['thumbnail']['src'], quote=True)}" class="img-thumbnail-custom publication-thumb" alt="{escape(publication['thumbnail']['alt'], quote=True)}">
        </div>
        <div class="col-md-8">
          <h3 class="publication-title">{escape(publication['title'])}</h3>
          <p class="publication-authors">{publication['authors_html']}</p>
          <p class="publication-venue"><em>{escape(publication['venue'])}, {publication['year']}</em></p>
          <p class="publication-links">
{links_html}
          </p>
        </div>
      </article>"""
        )
    return "\n\n".join(chunks)


def render_thesis_items(items: list[dict[str, Any]]) -> str:
    chunks = []
    for thesis in items:
        links_html = "\n".join(f"              {render_link(link)}" for link in thesis["links"])
        supervisors = ", ".join(escape(name) for name in thesis["supervisors"])
        chunks.append(
            f"""      <article class="row publication-card" id="thesis-{escape(thesis['slug'], quote=True)}">
        <div class="col-md-3">
          <img src="{escape(thesis['thumbnail']['src'], quote=True)}" class="img-thumbnail-logo thesis-thumb" alt="{escape(thesis['thumbnail']['alt'], quote=True)}">
        </div>
        <div class="col-md-8">
          <h3 class="publication-title">{escape(thesis['title'])}</h3>
          <p class="publication-subtitle">{escape(thesis['subtitle'])}</p>
          <p class="publication-authors">{escape(thesis['degree'])} <em>{escape(thesis['date'])}</em></p>
          <p class="publication-venue">Supervisor: {supervisors}</p>
          <p class="publication-links">
{links_html}
          </p>
        </div>
      </article>"""
        )
    return "\n\n".join(chunks)


def render_experience(items: list[dict[str, Any]]) -> str:
    chunks = []
    for item in items:
        logo = item.get("logo") or {}
        logo_html = ""
        if logo.get("src"):
            logo_html = (
                "            <div class=\"col-md-3\">"
                f"<img class=\"experience-logo img-thumbnail-logo\" src=\"{escape(logo['src'], quote=True)}\" alt=\"{escape(logo.get('alt', ''), quote=True)}\">"
                "</div>"
            )
        chunks.append(
            f"""        <li id="experience-{escape(item['slug'], quote=True)}">
          <p class="timeline-period">{escape(item['period'])}</p>
          <div class="row align-items-center experience-entry">
            <div class="col-md-8">
              <h3 class="experience-title">{escape(item['role'])} @ {item['organization_html']}</h3>
              <p class="experience-description">{escape(item['description'])}</p>
            </div>
{logo_html}
          </div>
        </li>"""
        )
    return "\n".join(chunks)


def render_projects(items: list[dict[str, Any]]) -> str:
    chunks = []
    for project in items:
        status_html = (
            f'<span class="project-status">{escape(project["status"])}</span>'
            if project.get("status")
            else ""
        )
        chunks.append(
            f"""        <article class="col-sm-6 col-lg-4 project-card" id="project-{escape(project['slug'], quote=True)}">
          <div class="project-card__inner">
            <img src="{escape(project['image']['src'], quote=True)}" class="img-thumbnail-custom project-thumb" alt="{escape(project['image']['alt'], quote=True)}">
            <h3 class="project-title"><a href="{escape(project['href'], quote=True)}" target="_blank" rel="noopener noreferrer">{escape(project['name'])}</a>{status_html}</h3>
            <p class="project-description">{escape(project['description'])}</p>
          </div>
        </article>"""
        )
    return "\n".join(chunks)


def render_misc(home: dict[str, Any]) -> str:
    chunks = []
    for heading, values in home["misc"].items():
        line = ", ".join(escape(value) for value in values)
        chunks.append(
            f'      <p class="text-large-2 misc-line"><strong>{escape(heading)}:</strong> {line}</p>'
        )
    return "\n".join(chunks)


def render_html(data: SiteData) -> str:
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    research_interests = ", ".join(data.home["profile"]["research_interests"])
    footer = data.site["footer"]
    rendered = template.substitute(
        TITLE=data.site["site"]["title"],
        DESCRIPTION=data.site["site"]["description"],
        GOOGLE_SITE_VERIFICATION=data.site["site"]["google_site_verification"],
        FAVICON=data.site["site"]["favicon"],
        BRAND=data.site["site"]["brand"],
        NAV_ITEMS=render_nav(data.site),
        PROFILE_IMAGE_SRC=data.home["profile"]["image"]["src"],
        PROFILE_IMAGE_ALT=data.home["profile"]["image"]["alt"],
        PROFILE_NAME=data.home["profile"]["name"],
        BIO_PARAGRAPHS=render_bio_paragraphs(data.home),
        RESEARCH_INTERESTS=escape(research_interests),
        CONTACT_LINKS=render_contact_links(data.site, data.home),
        PUBLICATIONS=render_publications(data.publications),
        THESES=render_thesis_items(data.thesis),
        EXPERIENCE=render_experience(data.experience),
        PROJECTS=render_projects(data.projects),
        MISC_ITEMS=render_misc(data.home),
        FOOTER_COPYRIGHT=f"Copyright © {footer['owner']} ({footer['start_year']}-{footer['end_year']})",
    )
    return "<!-- Generated by tools/site_cli.py. Do not edit this file directly. -->\n" + rendered + "\n"


def validate_non_empty(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")


def validate_unique_slugs(items: list[dict[str, Any]], name: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for item in items:
        slug = item.get("slug")
        validate_non_empty(slug, f"{name}.slug", errors)
        if isinstance(slug, str):
            if slug in seen:
                errors.append(f"duplicate slug in {name}: {slug}")
            seen.add(slug)


def validate_image(image: dict[str, Any], label: str, errors: list[str]) -> None:
    if not isinstance(image, dict):
        errors.append(f"{label} must be an object")
        return
    validate_non_empty(image.get("src"), f"{label}.src", errors)
    validate_non_empty(image.get("alt"), f"{label}.alt", errors)
    src = image.get("src")
    if isinstance(src, str) and not is_external(src) and not local_path(src).exists():
        errors.append(f"missing local asset for {label}.src: {src}")


def validate_optional_image(image: dict[str, Any] | None, label: str, errors: list[str]) -> None:
    if image is None:
        return
    if not isinstance(image, dict):
        errors.append(f"{label} must be an object when provided")
        return
    src = image.get("src")
    alt = image.get("alt")
    if not src and not alt:
        return
    validate_non_empty(src, f"{label}.src", errors)
    validate_non_empty(alt, f"{label}.alt", errors)
    if isinstance(src, str) and src and not is_external(src) and not local_path(src).exists():
        errors.append(f"missing local asset for {label}.src: {src}")


def validate_links(links: list[dict[str, Any]], label: str, errors: list[str]) -> None:
    if not isinstance(links, list) or not links:
        errors.append(f"{label} must contain at least one link")
        return
    for index, link in enumerate(links):
        prefix = f"{label}[{index}]"
        validate_non_empty(link.get("label"), f"{prefix}.label", errors)
        href = link.get("href")
        validate_non_empty(href, f"{prefix}.href", errors)
        if isinstance(href, str) and not is_external(href) and not local_path(href).exists():
            errors.append(f"missing local asset for {prefix}.href: {href}")


def validate_site_data(data: SiteData) -> list[str]:
    errors: list[str] = []

    validate_non_empty(data.site["site"].get("title"), "site.title", errors)
    validate_non_empty(data.site["site"].get("description"), "site.description", errors)
    validate_non_empty(data.site["site"].get("brand"), "site.brand", errors)
    validate_non_empty(data.site["site"].get("favicon"), "site.favicon", errors)
    if not local_path(data.site["site"]["favicon"]).exists():
        errors.append(f"missing favicon file: {data.site['site']['favicon']}")
    validate_non_empty(data.site["cv"].get("href"), "cv.href", errors)
    if not local_path(data.site["cv"]["href"]).exists():
        errors.append(f"missing CV file: {data.site['cv']['href']}")

    validate_image(data.home["profile"]["image"], "profile.image", errors)
    for index, paragraph in enumerate(data.home["profile"]["intro_paragraphs"]):
        validate_non_empty(paragraph, f"profile.intro_paragraphs[{index}]", errors)
    if not data.home["profile"]["research_interests"]:
        errors.append("profile.research_interests must not be empty")

    for index, link in enumerate(data.home["contact_links"]):
        validate_non_empty(link.get("label"), f"contact_links[{index}].label", errors)
        validate_non_empty(link.get("href"), f"contact_links[{index}].href", errors)

    validate_unique_slugs(data.publications, "publications", errors)
    validate_unique_slugs(data.thesis, "thesis", errors)
    validate_unique_slugs(data.experience, "experience", errors)
    validate_unique_slugs(data.projects, "projects", errors)

    for index, publication in enumerate(data.publications):
        prefix = f"publications[{index}]"
        validate_non_empty(publication.get("title"), f"{prefix}.title", errors)
        validate_non_empty(publication.get("authors_html"), f"{prefix}.authors_html", errors)
        validate_non_empty(publication.get("venue"), f"{prefix}.venue", errors)
        validate_image(publication.get("thumbnail", {}), f"{prefix}.thumbnail", errors)
        validate_links(publication.get("links", []), f"{prefix}.links", errors)

    for index, thesis in enumerate(data.thesis):
        prefix = f"thesis[{index}]"
        validate_non_empty(thesis.get("title"), f"{prefix}.title", errors)
        validate_non_empty(thesis.get("degree"), f"{prefix}.degree", errors)
        validate_image(thesis.get("thumbnail", {}), f"{prefix}.thumbnail", errors)
        validate_links(thesis.get("links", []), f"{prefix}.links", errors)

    for index, experience in enumerate(data.experience):
        prefix = f"experience[{index}]"
        validate_non_empty(experience.get("period"), f"{prefix}.period", errors)
        validate_non_empty(experience.get("role"), f"{prefix}.role", errors)
        validate_non_empty(experience.get("organization_html"), f"{prefix}.organization_html", errors)
        validate_non_empty(experience.get("description"), f"{prefix}.description", errors)
        validate_optional_image(experience.get("logo"), f"{prefix}.logo", errors)

    for index, project in enumerate(data.projects):
        prefix = f"projects[{index}]"
        validate_non_empty(project.get("name"), f"{prefix}.name", errors)
        validate_non_empty(project.get("href"), f"{prefix}.href", errors)
        validate_non_empty(project.get("description"), f"{prefix}.description", errors)
        validate_image(project.get("image", {}), f"{prefix}.image", errors)

    rendered = render_html(data)
    ids = re.findall(r'\bid="([^"]+)"', rendered)
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    for duplicate in duplicate_ids:
        errors.append(f"duplicate rendered id: {duplicate}")

    bad_blank_links = re.findall(r'<a\b[^>]*target="_blank"(?![^>]*rel="noopener noreferrer")', rendered)
    if bad_blank_links:
        errors.append("rendered output contains target=_blank links without rel=noopener noreferrer")

    return errors


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_build(_: argparse.Namespace) -> int:
    data = load_site_data()
    errors = validate_site_data(data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    INDEX_PATH.write_text(render_html(data), encoding="utf-8")
    print(f"Wrote {INDEX_PATH.relative_to(ROOT)}")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    data = load_site_data()
    errors = validate_site_data(data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("All checks passed.")
    return 0


def cmd_format(_: argparse.Namespace) -> int:
    for filename in (
        "site.json",
        "home.json",
        "publications.json",
        "thesis.json",
        "experience.json",
        "projects.json",
    ):
        path = CONTENT_DIR / filename
        write_json(path, load_json(path))
        print(f"Formatted {path.relative_to(ROOT)}")
    return 0


def cmd_new_publication(args: argparse.Namespace) -> int:
    path = CONTENT_DIR / "publications.json"
    publications = load_json(path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = args.slug or f"new-publication-{timestamp}"
    if any(publication.get("slug") == slug for publication in publications):
        print(f"ERROR: duplicate publication slug '{slug}'", file=sys.stderr)
        return 1
    stub = {
        "slug": slug,
        "title": args.title or "New Publication Title",
        "thumbnail": {
            "src": "publications/your-paper/thumbnail.png",
            "alt": "New publication thumbnail",
        },
        "authors_html": "<strong>Duan Gao</strong>",
        "venue": args.venue or "Conference or Journal",
        "year": args.year or datetime.now().year,
        "links": [
            {
                "label": "Author PDF",
                "href": "publications/your-paper/paper.pdf",
                "external": False,
            }
        ],
    }
    publications.append(stub)
    write_json(path, publications)
    print(f"Appended publication stub with slug '{slug}' to {path.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage and build the academic homepage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build index.html from structured content.")
    build_parser.set_defaults(func=cmd_build)

    check_parser = subparsers.add_parser("check", help="Validate structured content and rendered output.")
    check_parser.set_defaults(func=cmd_check)

    format_parser = subparsers.add_parser("format", help="Normalize JSON formatting in content files.")
    format_parser.set_defaults(func=cmd_format)

    new_parser = subparsers.add_parser("new", help="Append a content stub.")
    new_subparsers = new_parser.add_subparsers(dest="new_command", required=True)

    publication_parser = new_subparsers.add_parser("publication", help="Append a new publication stub.")
    publication_parser.add_argument("--slug", help="Slug for the publication entry.")
    publication_parser.add_argument("--title", help="Title for the publication entry.")
    publication_parser.add_argument("--venue", help="Venue for the publication entry.")
    publication_parser.add_argument("--year", type=int, help="Publication year.")
    publication_parser.set_defaults(func=cmd_new_publication)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
