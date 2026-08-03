# Simultaneous Localization and Mapping (SLAM)

## Setup

### Sharing locally connected USB devices with Docker containers in WSL2

Docker containers running via WSL2 do not have direct access to USB devices
connected to the Windows host by default. To make a USB device available inside
WSL2 (and optionally to Docker containers), use `usbipd-win`:

1. Install [usbipd-win](https://github.com/dorssel/usbipd-win) via Powershell

2. List devices: `usbipd list` (note `BUSID`)

3. Share device: `usbipd bind --busid=<BUSID>` (will persist across reboots; `STATE=Shared`)

4. Attach device to WSL2: `usbipd attach --wsl --busid=<BUSID>` (will not persist after WSL2 is shut down)

5. Confirm that WSL2 can see USB device: `lsusb` in WSL2 (should see device) and `usbipd list` in Powershell (`STATE=Attached`)

6. Find the USB device path in WSL2: `ls -l /dev/bus/usb/<BUS_NUM>/<DEVICE_NUM>`

7. Find `ID_SERIAL` property for device: `udevadm info --query=property --name=<device_path>`

8. Create stable link and mount in Docker [in progress]

```
Physical USB device
        |
        v
Windows USB stack
        |
        v
usbipd-win (shares device)
        |
        v
WSL2 Linux kernel (creates /dev nodes)
        |
        v
Docker container (--device=<path_in_WSL2>)
```

## Stereo SLAM