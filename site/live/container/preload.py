# What a kernel runs after the landing page's setup cell and before its page
# connects (server.py's PRELOAD, and warm.py): `pipe` reporting its steps.
import progress

pipe = progress.Reporting(pipe)  # noqa: F821 - the setup cell defines it
