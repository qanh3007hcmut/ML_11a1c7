import time
import threading
from datetime import datetime
import sys

class TimerLogger:
    """Class để log thời gian chạy real-time trong quá trình train, predict, test model."""
    
    def __init__(self, interval: int = 10, task_type: str = "Training"):
        """
        Args:
            interval (int): Số giây giữa các lần log (mặc định: 10 giây)
            task_type (str): Loại tác vụ ('Training', 'Predicting', 'Testing', ...)
        """
        self.interval = interval
        self.task_type = task_type  # Xác định nội dung log
        self.start_time = None
        self.stop_flag = False
        self.thread = None

    def _log_time(self):
        """Hàm chạy trong background để log thời gian trên cùng một dòng."""
        while not self.stop_flag:
            elapsed = int(time.time() - self.start_time)  # Làm tròn giây
            print(f"⏳ Training in progress... {elapsed:.2f} seconds elapsed", end="\r")
            
        elapsed = int(time.time() - self.start_time)
        sys.stdout.write(f"\r✅ {self.task_type} finished in {elapsed} seconds ({elapsed / 60:.2f} minutes)\n")
        sys.stdout.flush()

    def start(self):
        """Bắt đầu log thời gian."""
        self.start_time = time.time()
        print(f"🕒 {self.task_type} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.stop_flag = False
        self.thread = threading.Thread(target=self._log_time)
        self.thread.start()

    def stop(self):
        """Dừng log thời gian."""
        self.stop_flag = True
        if self.thread:
            self.thread.join()

    def set_task(self, task_type: str):
        """Thay đổi loại tác vụ đang thực hiện."""
        self.task_type = task_type
