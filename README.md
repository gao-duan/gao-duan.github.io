# gao-duan.github.io

Personal academic homepage published via GitHub Pages: https://gao-duan.github.io/

## Content workflow

The homepage is generated from structured content files under `content/`.

- `content/site.json`: site metadata, navigation, footer, CV link
- `content/home.json`: bio, contact links, misc section
- `content/publications.json`: publications list
- `content/thesis.json`: thesis entries
- `content/experience.json`: experience timeline
- `content/projects.json`: project cards

## CLI

All management commands are pure `Python 3` and use only the standard library.

```bash
python3 tools/site_cli.py check
python3 tools/site_cli.py build
python3 tools/site_cli.py format
python3 tools/site_cli.py new publication --slug my-paper
```

`index.html` is generated output. Edit the JSON files instead of modifying `index.html` directly.
