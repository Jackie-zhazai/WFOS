"""Keyword routing between Feature and Bugfix machines."""
from __future__ import annotations

from wfos.harness.router import route


def test_feature_keywords():
    assert route("新增一个用户模块，实现在 app.py 追加 feature_user()") == "feature"
    assert route("开发新模块: 支付回调") == "feature"
    assert route("需要实现一个导出功能") == "feature"
    assert route("加一个删除接口") == "feature"


def test_bugfix_keywords():
    assert route("系统报错：接口返回 500") == "bugfix"
    assert route("程序崩溃闪退") == "bugfix"
    assert route("运行结果不对，计算有 bug") == "bugfix"
    assert route("模块卡死没反应") == "bugfix"


def test_no_match():
    assert route("今天天气怎么样") is None
    assert route("") is None
    assert route("   ") is None


def test_tie_prefers_bugfix():
    # "实现" (feature) + "报错" (bugfix): equal weight -> diagnosis-first
    assert route("实现的功能报错了") == "bugfix"
