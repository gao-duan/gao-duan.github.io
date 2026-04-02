# gao-duan.github.io

Public deploy repository for `https://gao-duan.github.io/`.

This repository is intended to store generated output only:

- homepage publish files at the repository root
- blog publish files under `blog/`

Source content, templates, scripts, and local staging now live in the sibling source repository:

- `/Users/duangao/Projects/BlogCli`

Common publish commands:

```bash
python3 /Users/duangao/Projects/BlogCli/tools/publish_cli.py check-all
python3 /Users/duangao/Projects/BlogCli/tools/publish_cli.py build-all
python3 /Users/duangao/Projects/BlogCli/tools/publish_cli.py publish-all
```

After publishing, commit and push changes from this repository as usual.
