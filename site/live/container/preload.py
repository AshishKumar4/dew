# What a kernel runs after the landing page's setup cell and before its page
# connects (server.py's PRELOAD, and warm.py): the setup cell's loaders, each
# reporting a model's load, and what they return reporting their progress.
import progress

from_pretrained = progress.ReportingModels(from_pretrained, progress.Reporting)  # noqa: F821 - the setup cell defines it
text_model = progress.ReportingModels(text_model, progress.ReportingText)  # noqa: F821 - the setup cell defines it
pipe = from_pretrained("dewml/hybrid-dit-176m", revision="32d59de89683d59824361144b87bdcaf3e742598")
