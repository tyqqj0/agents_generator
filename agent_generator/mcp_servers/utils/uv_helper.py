# -*- coding: utf-8 -*-
"""
@File    :   uv_helper.py
@Time    :   2025/04/23 14:34:50
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





"""
UV路径辅助工具

用于获取UV环境中的命令路径
"""

import os
import sys
from pathlib import Path
from typing import Optional, List


class UVPathHelper:
    """UV路径辅助类，用于获取UV环境中的命令路径"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化UV路径辅助类

        Args:
            base_dir: UV虚拟环境目录，默认为上一级目录的uv_env
        """
        if base_dir is None:
            # 默认使用上一级目录的uv_env
            self.base_dir = (
                Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                / "uv_env"
            )
        else:
            self.base_dir = Path(base_dir)

        # 是否为Windows平台
        self.is_windows = sys.platform == "win32"

    def get_python_path(self) -> str:
        """获取Python解释器路径"""
        bin_dir = "Scripts" if self.is_windows else "bin"
        python_exe = "python.exe" if self.is_windows else "python"
        return str(self.base_dir / bin_dir / python_exe)

    def get_uv_path(self) -> str:
        """获取UV可执行文件路径"""
        bin_dir = "Scripts" if self.is_windows else "bin"
        uv_exe = "uv.exe" if self.is_windows else "uv"
        return str(self.base_dir / bin_dir / uv_exe)

    def exists(self) -> bool:
        """检查UV环境是否存在"""
        return self.base_dir.exists() and Path(self.get_python_path()).exists()

    @staticmethod
    def get_command_args(
        script_path: str, args: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取使用UV环境运行脚本的参数列表

        Args:
            script_path: 脚本路径
            args: 额外的参数

        Returns:
            参数列表
        """
        result = [
            "--directory",
            os.path.dirname(os.path.abspath(script_path)),
            "run",
            os.path.basename(script_path),
        ]
        if args:
            result.extend(args)
        return result


# 单例实例，便于导入使用
default_helper = UVPathHelper()

if __name__ == "__main__":
    helper = UVPathHelper()
    print(f"UV环境路径: {helper.base_dir}")
    print(f"Python路径: {helper.get_python_path()}")
    print(f"UV路径: {helper.get_uv_path()}")
    print(f"UV环境存在: {helper.exists()}")
