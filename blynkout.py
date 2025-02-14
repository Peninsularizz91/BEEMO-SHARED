import BlynkLib
import socket

# Your Blynk authentication token
BLYNK_AUTH = 'Lq0F3W3ovPyeXuqvtFInHhMrgUIzzs6K'

try:
    # Initialize Blynk with the correct server URL and SSL port
    blynk = BlynkLib.Blynk(BLYNK_AUTH, server='sgp1.blynk.cloud', port=443)
except (socket.gaierror, ValueError) as e:
    print(f"Failed to connect to Blynk server: {e}")

# Function to send a string to a virtual pin
def send_string_to_blynk(pin, string):
    try:
        blynk.virtual_write(pin, string)
        print(f"Sent '{string}' to virtual pin V{pin}")
    except Exception as e:
        print(f"Failed to send data to Blynk: {e}")

# Function to trigger a notification
def trigger_notification(event_name, event_code, description):
    try:
        print(f"Attempting to trigger notification: {event_name} - {description}")
        # Use log_event instead of notify
        blynk.log_event(event_code, f"{event_name}: {description}")
        print(f"Notification triggered: {event_name} - {description}")
    except Exception as e:
        print(f"Failed to trigger notification: {e}")

# Function to run Blynk
def run_blynk():
    while True:
        try:
            blynk.run()
        except Exception as e:
            print(f"Error while running Blynk: {e}")