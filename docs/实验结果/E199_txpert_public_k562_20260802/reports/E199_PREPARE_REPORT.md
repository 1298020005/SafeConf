# E199 prepare 报告

- 生成时间：`2026-08-02T01:22:27+08:00`
- TxPert commit：`08d82eea86746b044cf7531f4ec8c5f60e1cb73f`
- Zenodo record：`15420279`
- 输入资产：2 个官方 ZIP；只检查字节、哈希和 ZIP 目录，未执行模型。
- prepare gates：41/41
- 静态防泄漏检查：4/4
- 运行环境：`2.6.0+cu124 12.4 2.6.1 1.11.1 2.5.1 True`

下一步必须先把 runner、adapter、冻结合同和本 prepare 产物提交到 GitHub 与 Gitee，
确认三端 commit 完全相同，再解压资产并执行运行时 `batch.x` 置零不变性测试。任何
失败项都会阻止 formal；不会删除失败行后继续。
