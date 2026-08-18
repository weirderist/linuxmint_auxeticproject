"""
bot.py — DEPRECATED. Do not run this file.

This was an earlier, separate Telegram bot entrypoint, built independently
of server_bot.py (which is the one your systemd service actually runs --
confirmed via `ExecStart` in auxetic_bot.service).

Running BOTH bot.py and server_bot.py at the same time, using the same
TELEGRAM_BOT_TOKEN, is a real problem, not just redundancy: Telegram's
Bot API only allows one active long-polling connection per token. A
second instance polling with the same token causes repeated 409 Conflict
errors, and depending on timing, can cause commands to be silently
dropped or handled by whichever instance happens to win the race -- not
a visible crash, just inconsistent behaviour that's hard to diagnose
after the fact.

This file is kept only because "all 10 filenames, unchanged" was the
requirement -- it intentionally refuses to run, rather than silently
staying a second working bot that could get started by accident (e.g.
by a stray systemd unit, a forgotten terminal session, or a future
version of this project that adds a service for it without realizing
it duplicates server_bot.py).

If you want this bot's functionality, port anything genuinely missing
into server_bot.py's command set instead of running this file directly.
"""

import sys

if __name__ == "__main__":
    print(
        "bot.py is deprecated and will not start.\n"
        "server_bot.py is the active bot (see auxetic_bot.service).\n"
        "Running both with the same TELEGRAM_BOT_TOKEN causes Telegram "
        "API 409 Conflict errors from duplicate long-polling connections.\n"
        "If you specifically need to run this anyway, you understand the "
        "conflict risk above -- stop server_bot.py's systemd service first."
    )
    sys.exit(1)
