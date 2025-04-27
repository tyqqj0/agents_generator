# -*- coding: utf-8 -*-
"""
@File    :   uv_manager.py
@Time    :   2025/04/23 14:34:55
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





"""
UV虚拟环境管理工具

此模块提供创建和管理UV虚拟环境的工具。
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


class UVManager:
    """UV虚拟环境管理器"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化UV管理器

        参数:
            base_dir: UV虚拟环境基础目录，默认为mcp_servers目录下的uv_env
        """
        if base_dir is None:
            # 默认放在mcp_servers目录下
            self.base_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "uv_env"
        else:
            self.base_dir = Path(base_dir)

        # 确保目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 是否为Windows平台
        self.is_windows = sys.platform == "win32"

        # Python解释器路径
        self.python_path = self._get_python_path()

    def _get_python_path(self) -> str:
        """获取Python解释器路径"""
        bin_dir = "Scripts" if self.is_windows else "bin"
        python_exe = "python.exe" if self.is_windows else "python"
        return str(self.base_dir / bin_dir / python_exe)

    def is_uv_installed(self) -> bool:
        """检查UV是否已安装"""
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def install_uv(self) -> bool:
        """安装UV"""
        try:
            # 使用pip安装uv
            cmd = [sys.executable, "-m", "pip", "install", "uv"]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            print("安装UV失败，请手动安装: pip install uv")
            return False

    def create_environment(self) -> bool:
        """创建UV虚拟环境"""
        try:
            if not self.is_uv_installed():
                if not self.install_uv():
                    return False

            # 创建虚拟环境
            cmd = ["uv", "venv", str(self.base_dir)]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"创建UV虚拟环境失败: {e}")
            return False

    def install_packages(self, packages: List[str]) -> bool:
        """
        在UV虚拟环境中安装包

        参数:
            packages: 要安装的包列表

        返回:
            安装是否成功
        """
        try:
            if not os.path.exists(self.python_path):
                if not self.create_environment():
                    return False

            # 使用UV安装包
            cmd = ["uv", "pip", "install"]
            cmd.extend(packages)
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"安装包失败: {e}")
            return False

    def install_requirements(self, requirements_file: str) -> bool:
        """
        从requirements文件安装依赖

        参数:
            requirements_file: requirements文件路径

        返回:
            安装是否成功
        """
        try:
            if not os.path.exists(self.python_path):
                if not self.create_environment():
                    return False

            # 使用UV安装依赖
            cmd = ["uv", "pip", "install", "-r", requirements_file]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"安装依赖失败: {e}")
            return False

    def get_python_command(self) -> str:
        """获取Python解释器命令"""
        return self.python_path

    def run_script(self, script_path: str, args: Optional[List[str]] = None) -> int:
        """
        使用UV环境运行Python脚本

        参数:
            script_path: 脚本路径
            args: 脚本参数

        返回:
            脚本执行的返回码
        """
        if not os.path.exists(self.python_path):
            if not self.create_environment():
                return 1

        cmd = [self.python_path, script_path]
        if args:
            cmd.extend(args)

        result = subprocess.run(cmd)
        return result.returncode


if __name__ == "__main__":
    # 测试UV管理器
    manager = UVManager()
    print(f"UV已安装: {manager.is_uv_installed()}")
    print(f"Python路径: {manager.get_python_command()}")

    if not os.path.exists(manager.python_path):
        print("创建UV环境...")
        if manager.create_environment():
            print("UV环境创建成功")
        else:
            print("UV环境创建失败")

    # 安装基本依赖, 从requirements.txt安装
    manager.install_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
