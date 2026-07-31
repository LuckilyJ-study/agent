from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image


SERVICE_DIR = Path("/data_16T/deepseek/lzl/service")
EXTERNAL_OBS_DIR = Path("/home/v-wenhuitan/pi_0_open/media/obs").resolve()


if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))


_original_save = Image.Image.save


def _save_without_external_obs_write(self, fp, *args, **kwargs):
    if isinstance(fp, (str, Path)):
        target = Path(fp).expanduser().resolve()
        try:
            target.relative_to(EXTERNAL_OBS_DIR)
        except ValueError:
            pass
        else:
            print(f"[agent-master test] skipped external debug image write: {target}")
            return None
    return _original_save(self, fp, *args, **kwargs)


Image.Image.save = _save_without_external_obs_write

import pi05_service  # noqa: E402


if __name__ == "__main__":
    pi05_service.device, pi05_service.pi05 = pi05_service.start_service()
    port = int(os.getenv("PI05_TEST_PORT", "7777"))
    pi05_service.app.run(host="127.0.0.1", port=port)
