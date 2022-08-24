"""Handle configuration values.

Intended to be simple and read-only.

Values are read from a file (python data-type syntax) and then
not allowed to change afterwards. The "not allowed to change" is
weakly enforced.

The module global variable in this file is intended to be set once
in the main program:

import configs

if __name__ == "__main__":
  configs.cfg = configs.Configuration("/foo/bar/baz")

and then other code would treat the configs.cfg variable as a
read-only dictionary:

import configs

def foo():
  some_value = configs.cfg["some_default_value"]
  ... code that uses some_value ...

"""

import ast
import copy
import hashlib
import logging
import pprint

_LOGGER = logging.getLogger(__name__)


# Exceptions from this module
class ConfigException(Exception):
    pass


# Bad values passed to the class constructor
class BadInit(ConfigException):
    pass


# Attempt to set a configuration value
class ReadOnly(ConfigException):
    pass


class Configuration:
    def __init__(self, fname=None, defaults=None):
        """Initialize the configuration information."""
        if (defaults is not None) and (not isinstance(defaults, dict)):
            raise BadInit(
                "defaults is a %s, must be a dictionary: %s"
                % (type(defaults), defaults)
            )

        # It's a dictionary (or dictionary like object)
        if defaults:
            self._configs = defaults
        else:
            self._configs = {}

        if fname:
            with open(fname, "r") as f:
                data = f.read()
            from_file = ast.literal_eval(data)
            for k in from_file.keys():
                if k in self._configs.keys():
                    _LOGGER.warning(
                        "Overwriting key: %s (%s) with %s",
                        k,
                        self._configs[k],
                        from_file[k],
                    )
                self._configs[k] = from_file[k]

        if not self._configs:
            _LOGGER.warning("No values in config")

        _LOGGER.debug("configuration: %s", pprint.pformat(self._configs))

    def __getitem__(self, key):
        """Return a shallow copy of the configuration value.

        We do a shallow copy as a trade off between safety (caller can't
        modify) and time (could be a big list or dictionary).
        """
        return copy.copy(self._configs[key])

    def __setitem__(self, key, value):
        """We are treating this as a read-only config after it's been created."""
        raise ReadOnly("configuration values cannot be set: %s %s" % (key, value))

    def __len__(self):
        return len(self._configs)

    def keys(self):
        """Similar to the dictionary .keys()"""
        return self._configs.keys()

    def get(self, key, default=None):
        """Similar to the dictionary .get()"""
        return copy.copy(self._configs.get(key, default))

    def hash(self, ignore=None):
        """Creates a sha512 hash of the config.

        This hash may be useful when saving data for long periods of time. If the
        config hash has changed between when the data was saved and when it's used,
        it may be prudent to produce an error.

        Args:
          ignore: List of keys to ignore when hashing

        Returns: str. hexidecimal hash

        """
        if ignore is None:
            ignore = []
        filtered_config = {
            key: val for key, val in self._configs.items() if key not in ignore
        }
        return hashlib.sha512(str(filtered_config).encode()).hexdigest()


cfg = None
