"""Reading raw files, and one loader per dataset that maps it to the common schema.

``LOADERS`` names the loader module of every harmonised dataset; each module has
``plan(root, options)``, which lists the reading tasks of that dataset (one of
them, ``patients``, returns the static patients table). Scripts 02 and 06 use it.
"""

LOADERS = {
    "replace_bg": "t1dretrain.io.replace_bg",
    "dclp3": "t1dretrain.io.dclp3",
    "loop": "t1dretrain.io.loop",
    "hupa_ucm": "t1dretrain.io.hupa_ucm",
    "brist1d_open": "t1dretrain.io.brist1d",
    "t1d_uom": "t1dretrain.io.t1d_uom",
}
