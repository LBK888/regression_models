# -*- coding: utf-8 -*-
"""Language catalogues.

Each module here exports ``MESSAGES``: a mapping from the English source string
used in the code to the translated text. English needs no module, because the
source string is already the English text.

To add a language, create ``<code>.py`` with a ``MESSAGES`` dict and register
the code in :data:`i18n.LANGUAGES`. Any key left out falls back to English, so a
partial translation is usable rather than broken.
"""
