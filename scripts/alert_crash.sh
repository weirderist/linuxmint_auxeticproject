#!/bin/bash

# 1. Check for .env in project root or scripts folder
if [ -f "/home/sarvesh/auxetic_project/.env" ]; then
    ENV_PATH="/home/sarvesh/auxetic_project/.env"
elif [ -f "/home/sarvesh/auxetic_project/scripts/.env" ]; then
    ENV_PATH="/home/sarvesh/auxetic_project/scripts/.env"
fi

# 2. Export environment variables if found
if [ -n "$ENV_PATH" ]; then
    set -o allexport
    source "$ENV_PATH"
    set +o allexport
fi

CHAT_ID="8826140715"
MESSAGE="⚠️ CRASH ALERT: The telegram-bot daemon on KS-LinuxMintServer has unexpectedly crashed."

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${CHAT_ID}" \
    -d text="${MESSAGE}"
