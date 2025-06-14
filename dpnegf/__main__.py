from dpnegf.entrypoints.main import main as entry_main
import logging
import pyfiglet
from dpnegf import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def print_logo():
    f = pyfiglet.Figlet(font='starwars')  # 可选字体: 'standard', 'doom', 'starwars', 'slant'
    logo = f.renderText("DPNEGF").rstrip()

    banner_width = 80
    separator = "=" * banner_width

    log.info(separator)
    for line in logo.splitlines():
        log.info(line.center(banner_width))
    
    version_info = f"DPNEGF version {__version__}"
    log.info("-" * banner_width)
    log.info(version_info.center(banner_width))
    log.info(separator)
def main() -> None:
    """
    The main entry point for the dpnegf package.
    """
    print_logo()
    entry_main()

if __name__ == '__main__':
    main()
