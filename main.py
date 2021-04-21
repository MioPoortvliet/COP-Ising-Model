"""
Example script that simply calls an example. It ensures the root path is correct.
Author: Mio Poortvliet
"""
from examples.advanced_properties import main, settings

# Quick? Uncomment for quick but poor results (testing purposes only!):
from examples.advanced_properties import quick_settings as settings


if __name__ == '__main__':
	main(settings, "generated/data")
