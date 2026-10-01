"""Report the field each downloaded issue #18 wrapper config refuses.

PYTHONPATH=src python tools/wrapper_config_audit.py /path/to/config.json ...
Translation does not load weights and makes no claim about a storage codec.
"""

import argparse
import json
from pathlib import Path

from dew.interop.hf_decoders import translate_wrapper_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    observations = []
    for path in args.configs:
        config = json.loads(path.read_text())
        observation = {'path': str(path), 'model_type': config.get('model_type')}
        try:
            record = translate_wrapper_config(config)
        except ValueError as error:
            observation['refusal'] = str(error)
        else:
            observation['text_model_type'] = record['text_model_type']
            observation['loaded_config'] = True
        observations.append(observation)
    text = json.dumps(observations, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
