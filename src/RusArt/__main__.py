import sys
from streamlit.web import cli
from streamlit.web.cli import configurator_options, main
import streamlit
from pathlib import Path
import subprocess
from streamlit.web import bootstrap
import os

@main.command("hello")
@configurator_options
def main_hello(**kwargs):
    """Runs the Hello World script."""
    bootstrap.load_config_options(flag_options=kwargs)
    app_directory = Path(__file__).parent
    filename = str(Path(app_directory, 'RusArt.py'))
    cli._main_run(filename, flag_options=kwargs)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    kwargs = {"--server.enableXsrfProtection=false",
              "--server.enableCORS=true",
              "--server.port=3031",
              "--global.developmentMode=false"}
    sys.exit(main_hello(kwargs))
