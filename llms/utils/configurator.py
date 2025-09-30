"""
Poor Man's Configurator. Inherited from a different repo and kept out of simplicity 

Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

Note: it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

It's not a long term solution but works for the time being.
"""
import sys
from ast import literal_eval

# Set default config file
default_config = 'config/config.py'
config_file_loaded = False

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
        config_file_loaded = True
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

# If no config file was specified, use the default
if not config_file_loaded:
    print(f"No config file specified, using default: {default_config}")
    with open(default_config) as f:
        print(f.read())
    exec(open(default_config).read())