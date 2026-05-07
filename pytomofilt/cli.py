import typer

from .filter_models import ptf_reparam_filter_files
from .corr_and_spectra import calc_spectra, correlate


app = typer.Typer()
app.command()(ptf_reparam_filter_files)

app3 = typer.Typer()
app3.command()(calc_spectra)

app4 = typer.Typer()
app4.command()(correlate)


if __name__ == "__main__":
    app()