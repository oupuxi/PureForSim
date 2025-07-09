"""日志初始化"""
import logging

def setup_logger(name: str):
    """
    设置日志记录器。

    初始化一个可复用的模块日志器，它：
        拥有独立名字（模块名）
        设置为 DEBUG 级别
        输出到控制台
        带有时间、等级、模块名等格式
        自动避免重复添加 handler

    参数:
    name: str - 日志记录器的名称。

    返回:
    logger - 配置好的日志记录器对象。
    """
    # 创建或获取一个日志记录器
    logger = logging.getLogger(name)
    # 设置日志记录级别为DEBUG，记录所有级别的日志信息
    logger.setLevel(logging.DEBUG)

    # 检查日志记录器是否已经有处理程序，如果没有则进行配置
    if not logger.handlers:  # 防止重复添加
        # 创建一个控制台日志处理程序
        ch = logging.StreamHandler()
        # 设置处理程序的日志级别为DEBUG
        ch.setLevel(logging.DEBUG)

        # 创建一个日志格式器，指定日志信息的输出格式
        # - `%(asctime)s`：当前时间
        # - `%(levelname)s`：日志级别（INFO, ERROR 等）
        # - `%(name)s`：日志器的名字（即你传入的 `name`）
        # - `%(message)s`：日志内容
        # - `datefmt` 是时间格式，如 `2025-07-04 15:30:00`
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # 将格式器应用到处理程序上
        ch.setFormatter(formatter)
        # 将处理程序添加到日志记录器上
        logger.addHandler(ch)

    # 返回配置好的日志记录器
    return logger

"""
在每个模块：
    from mypackage.logger import setup_logger

    logger = setup_logger(__name__)
    logger.info("开始执行任务")
"""