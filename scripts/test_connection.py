import pycozmo
import time

with pycozmo.connect() as cli:
    print("Connected to Cozmo!")
    cli.drive_wheels(lwheel_speed=50, rwheel_speed=50)  # Test drive
    time.sleep(2)
    cli.drive_wheels(lwheel_speed=0, rwheel_speed=0)