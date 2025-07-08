import pytest


from src.logger.logger import setup_logger

logger = setup_logger(__name__)

def test_demo():
    logger.info("开始运行 demo")
    assert 1 == 1


