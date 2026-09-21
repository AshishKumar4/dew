"""One module per decoder family, read by the family table in `hf_decoders`.

A family module holds what is true of that family alone: its config
translation, its tensor path map, the fields its export writes and the
tensors it prepares. The table that names them is in
`dew.interop.hf_decoders`, which imports these modules below the shared
readers they call, so the hub is complete whichever module is reached first.
"""
