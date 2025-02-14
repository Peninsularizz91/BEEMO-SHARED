from datetime import datetime, time, timedelta
import threading
import time as time_module

# Default schedule times
START_TIME = time(6, 0)  # 6:00 AM
END_TIME = time(18, 0)   # 6:00 PM
DURATION_DAYS = 30       # Default duration: 30 days
INTERVAL = 3600          # Default interval: 1 hour (3600 seconds)

def is_within_schedule(current_time, start_time, end_time):
    return start_time <= current_time <= end_time

def should_run_today(current_date, duration_days):
    end_date = current_date + timedelta(days=duration_days)
    return current_date <= end_date

def schedule_task(task, start_time, end_time, interval, duration_days):
    current_date = datetime.now().date()

    if should_run_today(current_date, duration_days):
        next_run = datetime.combine(current_date, start_time)
        while next_run <= datetime.combine(current_date, end_time):
            now = datetime.now()
            if now >= next_run:
                if is_within_schedule(now.time(), start_time, end_time):
                    task()  # Run the task
                next_run += timedelta(seconds=interval)
            time_module.sleep(10)
    else:
        return

# Example task function
def record_audio_task():
    print("Recording audio...")

# Function to start the scheduled task
def start_scheduled_task():
    schedule_thread = threading.Thread(target=schedule_task, args=(record_audio_task, START_TIME, END_TIME, INTERVAL, DURATION_DAYS))
    schedule_thread.start()
