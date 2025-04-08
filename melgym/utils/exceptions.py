from colorama import Fore, Style

class MelgymError(Exception):
    """
    Custom exception for errors.
    """
    def __init__(self, message):
        super().__init__(message)
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")

class MelgymWarning(Warning):
    """
    Custom exception for warnings.
    """
    def __init__(self, message):
        super().__init__(message)
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")