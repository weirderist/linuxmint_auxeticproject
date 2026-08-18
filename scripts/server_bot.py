import os
import sys
import glob
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from sys_monitor import get_server_status, log_action
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set.")
    sys.exit(1)

ALLOWED_CHAT_ID = 8826140715

BASE_DIR = "/home/sarvesh/auxetic_project"
SIM_INPUT_DIR = os.path.join(BASE_DIR, "input_grids")
SIM_OUTPUT_DIR = os.path.join(BASE_DIR, "debug_output")
SHARED_DIR = os.path.join(BASE_DIR, "shared")
MATRIX_DIR = os.path.join(BASE_DIR, "output_matrices")
SOLVER_OUTPUT_DIR = os.path.join(BASE_DIR, "sim_output")

PYTHON_EXEC = os.path.join(BASE_DIR, "scripts/venv/bin/python")
CROP_SCRIPT = os.path.join(BASE_DIR, "scripts/crop_classify.py")
MAIN_SCRIPT = os.path.join(BASE_DIR, "scripts/main.py")

os.makedirs(SIM_INPUT_DIR, exist_ok=True)
os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)

# Tracks the most recently produced matrix file for this (single-user)
# bot, so /simulate can default to "whatever was just classified"
# without the user having to repeat the filename.
_last_matrix_path = {"path": None}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    log_action(update.effective_chat.id, "COMMAND: /start")
    await update.message.reply_text(
        "KS-LINUXMINTSERVER CONTROL\n\n"
        "Use /commands to view available operations."
    )


async def show_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    log_action(update.effective_chat.id, "COMMAND: /commands")
    cmd_text = (
        "*AVAILABLE SERVER COMMANDS*\n\n"
        "/status - Display real-time hardware telemetry and uptime\n"
        "/commands - Show this reference list\n"
        "/analyse_pattern - Process an image through the grid classifier\n"
        "/simulate <cpi> <wpi> <loop_length_mm> - Run the puckering solver "
        "on the most recently classified matrix and render 4+4 views\n"
        "/save_shared - Store uploaded image or file directly into the shared folder\n\n"
        "*USAGE INSTRUCTIONS*\n"
        "- Send photo/document with caption `/analyse_pattern [rows] [cols]` to classify a grid\n"
        "- Then send `/simulate 14 18 3.2` to run the solver on that matrix (Ne fixed at 10, cotton)\n"
        "- Send photo or file with caption `/save_shared` to retain in shared directory"
    )
    await update.message.reply_text(cmd_text, parse_mode="Markdown")


async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    log_action(update.effective_chat.id, "COMMAND: /status")
    status_report = get_server_status()
    await update.message.reply_text(status_report, parse_mode="Markdown")


async def cmd_analyse_pattern_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    await update.message.reply_text(
        "USAGE: Attach a grid image and set the caption to `/analyse_pattern` "
        "(optionally follow with row col dimensions, e.g. `/analyse_pattern 12 16`).",
        parse_mode="Markdown"
    )


async def cmd_save_shared_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return
    await update.message.reply_text(
        "USAGE: Attach any file or photo and set the caption to `/save_shared` "
        "to store it in `~/auxetic_project/shared/`.",
        parse_mode="Markdown"
    )


async def process_simulation_image(update: Update, file_id, caption):
    file = await update.get_bot().get_file(file_id)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}.png"
    input_path = os.path.join(SIM_INPUT_DIR, filename)
    await file.download_to_drive(input_path)

    log_action(update.effective_chat.id, f"SIMULATION_INPUT: {filename}")

    parts = caption.replace("/analyse_pattern", "").strip().split()
    cmd = [PYTHON_EXEC, CROP_SCRIPT, filename]

    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        cmd.extend([parts[0], parts[1]])

    await update.message.reply_text(f"Processing grid classification `{filename}`...", parse_mode="Markdown")

    result = subprocess.run(cmd, capture_output=True, text=True)

    output_text = result.stdout if result.stdout else ""
    if result.stderr:
        output_text += f"\n{result.stderr}"
    if len(output_text) > 3000:
        output_text = output_text[-3000:]

    if result.returncode == 0:
        log_action(update.effective_chat.id, f"CLASSIFY_SUCCESS: {filename}")
        await update.message.reply_text(f"[SUCCESS]\n```\n{output_text}\n```", parse_mode="Markdown")
        matrix_path = os.path.join(MATRIX_DIR, os.path.splitext(filename)[0] + "_matrix.txt")
        if os.path.exists(matrix_path):
            _last_matrix_path["path"] = matrix_path
            await update.message.reply_text(
                f"Matrix saved. Run `/simulate <cpi> <wpi> <loop_length_mm>` "
                f"to generate the puckering prediction from this matrix.",
                parse_mode="Markdown"
            )
    else:
        log_action(update.effective_chat.id, f"CLASSIFY_FAILURE: Exit code {result.returncode}")
        await update.message.reply_text(f"[FAILED - Exit Code {result.returncode}]\n```\n{output_text}\n```", parse_mode="Markdown")

    debug_filename = os.path.splitext(filename)[0] + "_debug.png"
    debug_path = os.path.join(SIM_OUTPUT_DIR, debug_filename)
    if os.path.exists(debug_path):
        with open(debug_path, "rb") as debug_img:
            await update.message.reply_photo(photo=debug_img, caption="Visual Alignment Overlay")


async def cmd_simulate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return

    text_parts = update.message.text.replace("/simulate", "").strip().split()
    if len(text_parts) < 3:
        await update.message.reply_text(
            "USAGE: `/simulate <cpi> <wpi> <loop_length_mm>`\n"
            "Example: `/simulate 14 18 3.2`\n"
            "Runs on the most recently classified matrix. Ne is fixed at 10 (cotton).",
            parse_mode="Markdown"
        )
        return

    try:
        cpi, wpi, loop_length_mm = float(text_parts[0]), float(text_parts[1]), float(text_parts[2])
    except ValueError:
        await update.message.reply_text("cpi, wpi, and loop_length_mm must all be numbers.")
        return

    matrix_path = _last_matrix_path["path"]
    if matrix_path is None:
        candidates = sorted(glob.glob(os.path.join(MATRIX_DIR, "*_matrix.txt")))
        if not candidates:
            await update.message.reply_text(
                "No matrix found. Run /analyse_pattern on a grid image first."
            )
            return
        matrix_path = candidates[-1]

    log_action(update.effective_chat.id, f"SIMULATE: {matrix_path} cpi={cpi} wpi={wpi} l={loop_length_mm}")
    await update.message.reply_text(
        f"Running solver on `{os.path.basename(matrix_path)}` "
        f"(cpi={cpi}, wpi={wpi}, loop_length={loop_length_mm}mm)...",
        parse_mode="Markdown"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SOLVER_OUTPUT_DIR, run_id)

    cmd = [
        PYTHON_EXEC, MAIN_SCRIPT,
        "--matrix", matrix_path,
        "--cpi", str(cpi),
        "--wpi", str(wpi),
        "--loop-length-mm", str(loop_length_mm),
        "--output-dir", output_dir,
        "--render",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output_text = result.stdout if result.stdout else ""
    if result.stderr:
        output_text += f"\n{result.stderr}"
    if len(output_text) > 3500:
        output_text = output_text[-3500:]

    if result.returncode != 0:
        log_action(update.effective_chat.id, f"SIMULATE_FAILURE: exit {result.returncode}")
        await update.message.reply_text(f"[SOLVER FAILED]\n```\n{output_text}\n```", parse_mode="Markdown")
        return

    log_action(update.effective_chat.id, "SIMULATE_SUCCESS")
    await update.message.reply_text(f"[SOLVER OUTPUT]\n```\n{output_text}\n```", parse_mode="Markdown")

    summary_path = os.path.join(output_dir, "calculation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            await update.message.reply_text(f"```\n{f.read()}\n```", parse_mode="Markdown")

    sent_any_image = False
    for label, subdir in [("BEFORE", "views_before"), ("AFTER", "views_after")]:
        view_dir = os.path.join(output_dir, subdir)
        for img_path in sorted(glob.glob(os.path.join(view_dir, "*.png"))):
            with open(img_path, "rb") as img_f:
                await update.message.reply_photo(photo=img_f, caption=f"{label}: {os.path.basename(img_path)}")
            sent_any_image = True

    if not sent_any_image:
        await update.message.reply_text(
            "No rendered images found -- render_turntable_views requires Blender "
            "on PATH and hasn't been smoke-tested end to end yet. Check the "
            "solver output above for errors, and the .obj files in "
            f"`{output_dir}` are still usable directly in Blender if the "
            "render step failed.",
            parse_mode="Markdown"
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return

    caption = (update.message.caption or "").strip()
    photo = update.message.photo[-1]

    if caption.startswith("/save_shared"):
        file = await context.bot.get_file(photo.file_id)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shared_{timestamp_str}.png"
        dest_path = os.path.join(SHARED_DIR, filename)
        await file.download_to_drive(dest_path)
        log_action(update.effective_chat.id, f"SHARED_PHOTO_UPLOAD: {filename}")
        await update.message.reply_text(f"Photo saved to shared directory:\n`{dest_path}`", parse_mode="Markdown")
        return

    await process_simulation_image(update, photo.file_id, caption)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        return

    doc = update.message.document
    caption = (update.message.caption or "").strip()

    if caption.startswith("/analyse_pattern"):
        await process_simulation_image(update, doc.file_id, caption)
        return

    file = await context.bot.get_file(doc.file_id)
    dest_path = os.path.join(SHARED_DIR, doc.file_name)
    await file.download_to_drive(dest_path)

    log_action(update.effective_chat.id, f"SHARED_UPLOAD: {doc.file_name}")
    await update.message.reply_text(f"File saved to shared directory:\n`{dest_path}`", parse_mode="Markdown")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("commands", show_commands))
    app.add_handler(CommandHandler("status", show_status))
    app.add_handler(CommandHandler("analyse_pattern", cmd_analyse_pattern_info))
    app.add_handler(CommandHandler("save_shared", cmd_save_shared_info))
    app.add_handler(CommandHandler("simulate", cmd_simulate))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("KS-LINUXMINTSERVER listener running...")
    app.run_polling()


if __name__ == "__main__":
    main()
