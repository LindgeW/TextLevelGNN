import logging
import sys
import os

# 日志级别关系映射
level_dict = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}

LOG_PATH = "run.log"
LOG_FMT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S %p"
LOG_LEVEL = "debug"
# ENABLE_LOG = true
TO_CONSOLE = True
TO_FILE = False

loggers = dict()


def get_logger(name=None):
    global loggers

    if name is None:
        name = __name__

    if loggers.get(name):
        return loggers.get(name)

    # 设置日志输出格式
    fmt = logging.Formatter(fmt=LOG_FMT,
                            datefmt=DATE_FMT)
    # 创建一个名为filename的日志器
    logger = logging.getLogger(LOG_PATH)
    # 设置日志级别
    logger.setLevel(level_dict[LOG_LEVEL])

    if TO_CONSOLE:
        # 获取控制台输出的处理器
        console_handler = logging.StreamHandler(sys.stdout)  # 默认是sys.stderr
        # 设置控制台处理器的等级为DEBUG
        console_handler.setLevel(level_dict[LOG_LEVEL])
        # 设置控制台输出日志的格式
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    if TO_FILE:
        # 获取路径的目录
        log_dir = os.path.dirname(LOG_PATH)
        if os.path.isdir(log_dir) and not os.path.exists(log_dir):
            # 目录不存在则创建
            os.makedirs(log_dir)

        # 获取文件输出的处理器
        file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
        # 设置文件输出处理器的等级为INFO
        file_handler.setLevel(level_dict[LOG_LEVEL])
        # 设置文件输出日志的格式
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    loggers[name] = logger
    return logger


logger = get_logger()
