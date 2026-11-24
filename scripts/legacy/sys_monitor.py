import os
import psutil
import time
import datetime

LOG_DIR = "/home/sarvesh/auxetic_project/logs"

def log_action(user_id, action):
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    log_file = os.path.join(LOG_DIR, f"{today}.log")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] USER:{user_id} | ACTION: {action}\n")

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps and temps['coretemp']:
            return f"{temps['coretemp'][0].current:.1f}°C"
        if 'k10temp' in temps and temps['k10temp']:
            return f"{temps['k10temp'][0].current:.1f}°C"
        if 'nvme' in temps and temps['nvme']:
            return f"{temps['nvme'][0].current:.1f}°C"
        for name, entries in temps.items():
            if name != 'acpitz' and entries:
                return f"{entries[0].current:.1f}°C"
    except Exception:
        pass
    return "N/A"

def get_server_status():
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp_str = get_cpu_temp()
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    ram_used = f"{ram.used / (1024**3):.1f}GB"
    ram_total = f"{ram.total / (1024**3):.1f}GB"
    disk = psutil.disk_usage('/')
    disk_total = f"{disk.total / (1024**3):.1f}GB"
    disk_free = f"{disk.free / (1024**3):.1f}GB"

    status_msg = (
        "*KS-LINUXMINTSERVER STATUS*\n\n"
        f"*Date & Time*: {current_time_str}\n"
        f"*Uptime*: {uptime_str}\n"
        f"*CPU Temp*: {temp_str}\n"
        f"*CPU Load*: {cpu_percent}%\n"
        f"*RAM Usage*: {ram_used} / {ram_total} ({ram.percent}%)\n"
        f"*Disk (Root)*: {disk_free} free of {disk_total} ({disk.percent}%)\n"
    )
    return status_msg
