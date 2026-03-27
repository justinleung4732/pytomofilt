import typer

from .filter_models import ptf_reparam_filter_files
from .bg_spike import resolution_test_bg_spike


app = typer.Typer()
app.command()(ptf_reparam_filter_files)

app2 = typer.Typer()
app2.command()(resolution_test_bg_spike)


if __name__ == "__main__":
    app()