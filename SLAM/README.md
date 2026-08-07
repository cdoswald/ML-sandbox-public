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

6. Find the available video devices: `ls -al /dev/video*`

7. Identify video feed vs control nodes: `udevadm info --query=property --name=/dev/video<#>` (video feed will have `ID_V4L_CAPABILITIES=:capture:`)

8. Mount the video feeds in docker-compose file under `<service>.devices`.

9. Test video streaming with Video4Linux2: `v4l2-ctl --verbose --stream-mmap --stream-count=100 --stream-to=frame.raw`. If frames are dropped, you can try reducing the width/height with `v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG` and frame rate with `v4l2-ctl --set-parm=15`.

10. If you're still getting errors for most or all frames at the desired resolution and frame rate, it may be the case that USB/IP can't transfer frames reliably (since webcam is probably isochronous USB transfer, i.e., there are no error retries). In that case, you may need to switch to a native Linux machine rather than using WSL2 with `usbipd-win`.

```
USB webcam
   |
USB cable
   |
Windows USB host controller
   |
usbipd-win
   |
WSL2 USB/IP kernel driver
   |
Linux uvcvideo
   |
V4L2
```

## Stereo SLAM