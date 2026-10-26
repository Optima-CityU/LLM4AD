# SPDX-License-Identifier: BSD-2-Clause
"""Inspect built artifacts for required configs, versions and license notices."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    wheels = list(args.directory.glob('*.whl'))
    sources = list(args.directory.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit('Expected one wheel and one sdist')
    version = runpy.run_path(str(ROOT / 'llm4ad/_version.py'))['__version__']
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        for path in (ROOT / 'llm4ad').rglob('*.yaml'):
            if path.relative_to(ROOT).as_posix() not in names:
                raise SystemExit(f'Missing wheel resource: {path}')
        required = ['llm4ad/task/catalogue.json', 'llm4ad/tools/llm/replay.py',
                    'llm4ad/base/errors.py', 'llm4ad/method/_runtime.py']
        if not all(name in names for name in required):
            raise SystemExit('Missing catalogue or replay module')
        metadata = archive.read(next(n for n in names if n.endswith('/METADATA'))).decode()
        if f'Version: {version}\n' not in metadata:
            raise SystemExit('Version metadata mismatch')
        if 'Project-URL: Source, https://github.com/Optima-CityU/LLM4AD\n' not in metadata:
            raise SystemExit('Missing canonical source URL')
        for suffix in ['/licenses/LICENSE', '/licenses/LICENSES/Apache-2.0.txt', '/licenses/NOTICE']:
            if not any(name.endswith(suffix) for name in names):
                raise SystemExit(f'Missing license: {suffix}')
        if any(name.endswith('.pyc') or '/.git/' in name or name.endswith('.DS_Store') for name in names):
            raise SystemExit('Unexpected cache or VCS files in wheel')
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        for suffix in ['/tests/test_reproducibility.py', '/docs/index.md', '/SECURITY.md',
                       '/LICENSE', '/LICENSES/Apache-2.0.txt', '/NOTICE',
                       '/THIRD_PARTY_NOTICES.md', '/llm4ad/task/catalogue.json',
                       '/tests/test_search_contracts.py', '/tests/test_checkpoint_restoration.py',
                       '/.github/workflows/ci.yml', '/CHANGELOG.md']:
            if not any(name.endswith(suffix) for name in names):
                raise SystemExit(f'Missing source distribution file: {suffix}')
        if any(name.endswith('.pyc') or '/.git/' in name or name.endswith('.DS_Store') for name in names):
            raise SystemExit('Unexpected cache or VCS files in source distribution')
    print(f'Distribution resources, metadata and notices verified for {version}')


if __name__ == '__main__':
    main()
