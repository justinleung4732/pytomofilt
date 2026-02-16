import typer

from .filter_models import ptf_reparam_filter_files


app = typer.Typer()
app.command()(ptf_reparam_filter_files)


if __name__ == "__main__":
    app()