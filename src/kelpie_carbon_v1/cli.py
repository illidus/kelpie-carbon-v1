"""CLI module for Kelpie-Carbon v1."""
import typer

app = typer.Typer()


@app.command()
def hello():
    """Say hello from Kelpie-Carbon v1."""
    print("Kelpie-Carbon v1 lives!")


if __name__ == "__main__":
    app()
